import Foundation
import Metal
import MetalPerformanceShaders

public final class KVCraftVaeDecoder {
    public let context: MetalContext
    public let weights: WeightArchive
    public let latentHeight: Int
    public let latentWidth: Int

    private let ops: MetalOps
    private let skipAttentionForProfiling: Bool
    private let usePhaseUpsample: Bool
    private let steadyGraphExecutor: SteadyStateGraphExecutor?
    private var caches: [String: GpuTensor] = [:]
    private var validCaches: Set<String> = []

    public init(weightsURL: URL, latentHeight: Int = 45, latentWidth: Int = 80) throws {
        self.context = try MetalContext()
        self.weights = try WeightArchive(url: weightsURL, context: context)
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.ops = MetalOps(context: context)
        self.skipAttentionForProfiling = ProcessInfo.processInfo.environment["SOLARIS_SKIP_ATTENTION"] == "1"
        self.usePhaseUpsample = ProcessInfo.processInfo.environment["SOLARIS_PHASE_UPSAMPLE"] == "1"
        if ProcessInfo.processInfo.environment["SOLARIS_DISABLE_STEADY_GRAPH"] != "1" {
            self.steadyGraphExecutor = SteadyStateGraphExecutor(
                context: context,
                weights: weights,
                latentHeight: latentHeight,
                latentWidth: latentWidth,
                skipAttention: skipAttentionForProfiling
            )
        } else {
            self.steadyGraphExecutor = nil
        }
    }

    public func reset() {
        caches.removeAll()
        validCaches.removeAll()
    }

    public func makeLatentTensor(bytes: Data) throws -> GpuTensor {
        let shape = TensorShape(1, 1, latentHeight, latentWidth, 16)
        guard bytes.count == shape.byteCountF16 else {
            throw KVCraftMetalError.invalidArgument("latent payload is \(bytes.count) bytes; expected \(shape.byteCountF16)")
        }
        guard let buffer = context.device.makeBuffer(bytes: [UInt8](bytes), length: bytes.count, options: .storageModeShared) else {
            throw KVCraftMetalError.allocationFailed("latent input")
        }
        return GpuTensor(buffer: buffer, shape: shape)
    }

    @discardableResult
    public func decode(latent: GpuTensor, waitUntilCompleted: Bool = true) throws -> GpuTensor {
        guard let rawCommandBuffer = context.queue.makeCommandBuffer() else {
            throw KVCraftMetalError.allocationFailed("command buffer")
        }
        let commandBuffer = MPSCommandBuffer(commandBuffer: rawCommandBuffer)
        let out = try encodeDecode(latent: latent, commandBuffer: commandBuffer)
        commandBuffer.commit()
        if waitUntilCompleted {
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw error
            }
        }
        return out
    }

    @discardableResult
    public func encodeDecode(latent: GpuTensor, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        precondition(latent.shape.c == 16)
        if let steadyGraphExecutor, steadyGraphExecutor.canRun(validCaches: validCaches) {
            let oldCaches = caches
            let result = try steadyGraphExecutor.encode(latent: latent, caches: caches, commandBuffer: commandBuffer)
            for (name, tensor) in result.updatedCaches {
                caches[name] = tensor
                validCaches.insert(name)
            }
            let retained = Array(oldCaches.values) + result.retainedInputs
            commandBuffer.addCompletedHandler { _ in
                _ = retained
            }
            return result.output
        }
        var x = try ops.scaleLatent(
            latent,
            mean: weightsBuffer("vae_scale.mean"),
            std: weightsBuffer("vae_scale.std"),
            commandBuffer: commandBuffer
        )
        x = try convNoCache(x, "conv2", outChannels: 16, kernel: (1, 1, 1), pad: (0, 0, 0), commandBuffer: commandBuffer)
        return try decodeCore(x, commandBuffer: commandBuffer)
    }

    private func decodeCore(_ input: GpuTensor, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        var x = try causalConv(input, "decoder.conv1", outChannels: 384, kernel: (3, 3, 3), pad: (2, 1, 1), commandBuffer: commandBuffer)

        x = try residual(x, "decoder.middle.0", outChannels: 384, commandBuffer: commandBuffer)
        x = try attention(x, "decoder.middle.1", channels: 384, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.middle.2", outChannels: 384, commandBuffer: commandBuffer)

        x = try residual(x, "decoder.upsamples.0", outChannels: 384, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.1", outChannels: 384, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.2", outChannels: 384, commandBuffer: commandBuffer)
        x = try upsample2d(x, "decoder.upsamples.3", outChannels: 192, commandBuffer: commandBuffer)

        x = try residual(x, "decoder.upsamples.4", outChannels: 384, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.5", outChannels: 384, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.6", outChannels: 384, commandBuffer: commandBuffer)
        x = try upsample3d(x, "decoder.upsamples.7", outChannels: 192, commandBuffer: commandBuffer)

        x = try residual(x, "decoder.upsamples.8", outChannels: 192, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.9", outChannels: 192, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.10", outChannels: 192, commandBuffer: commandBuffer)
        x = try upsample3d(x, "decoder.upsamples.11", outChannels: 96, commandBuffer: commandBuffer)

        x = try residual(x, "decoder.upsamples.12", outChannels: 96, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.13", outChannels: 96, commandBuffer: commandBuffer)
        x = try residual(x, "decoder.upsamples.14", outChannels: 96, commandBuffer: commandBuffer)

        let (headBias, headHasBias) = weights.bias("decoder.head.0.bias")
        x = try ops.rmsNormSilu(
            x,
            gamma: weightsBuffer("decoder.head.0.gamma"),
            bias: headBias,
            hasBias: headHasBias,
            applySilu: true,
            commandBuffer: commandBuffer
        )
        x = try causalConv(x, "decoder.head.2", outChannels: 3, kernel: (3, 3, 3), pad: (2, 1, 1), commandBuffer: commandBuffer)
        return x
    }

    private func residual(_ input: GpuTensor, _ base: String, outChannels: Int, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        let shortcut: GpuTensor
        if input.shape.c == outChannels {
            shortcut = input
        } else {
            shortcut = try convNoCache(input, "\(base).shortcut", outChannels: outChannels, kernel: (1, 1, 1), pad: (0, 0, 0), commandBuffer: commandBuffer)
        }

        var x = try normSilu(input, "\(base).residual.0", commandBuffer: commandBuffer)
        x = try causalConv(x, "\(base).residual.2", outChannels: outChannels, kernel: (3, 3, 3), pad: (2, 1, 1), commandBuffer: commandBuffer)
        x = try normSilu(x, "\(base).residual.3", commandBuffer: commandBuffer)
        x = try causalConv(x, "\(base).residual.6", outChannels: outChannels, kernel: (3, 3, 3), pad: (2, 1, 1), commandBuffer: commandBuffer)
        return try ops.add(x, shortcut, commandBuffer: commandBuffer)
    }

    private func attention(_ input: GpuTensor, _ base: String, channels: Int, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        if skipAttentionForProfiling {
            return input
        }
        let (normBias, normHasBias) = weights.bias("\(base).norm.bias")
        let normed = try ops.rmsNormSilu(
            input,
            gamma: weightsBuffer("\(base).norm.gamma"),
            bias: normBias,
            hasBias: normHasBias,
            applySilu: false,
            commandBuffer: commandBuffer
        )
        let qkv = try convNoCache(normed, "\(base).to_qkv", outChannels: channels * 3, kernel: (1, 1, 1), pad: (0, 0, 0), commandBuffer: commandBuffer)
        let attended = try ops.attentionTiled(qkv, channels: channels, commandBuffer: commandBuffer)
        let projected = try convNoCache(attended, "\(base).proj", outChannels: channels, kernel: (1, 1, 1), pad: (0, 0, 0), commandBuffer: commandBuffer)
        return try ops.add(projected, input, commandBuffer: commandBuffer)
    }

    private func upsample2d(_ input: GpuTensor, _ base: String, outChannels: Int, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        let (bias, hasBias) = weights.bias("\(base).conv.bias")
        if usePhaseUpsample {
            return try ops.phaseUpsample2dConv(
                input,
                foldedWeight: weights.foldedNearest2xPhaseWeight("\(base).conv.kernel"),
                bias: bias,
                hasBias: hasBias,
                outChannels: outChannels,
                commandBuffer: commandBuffer
            )
        }
        return try ops.upsample2dConv(
            input,
            weight: weightsBuffer("\(base).conv.kernel"),
            bias: bias,
            hasBias: hasBias,
            outChannels: outChannels,
            commandBuffer: commandBuffer
        )
    }

    private func upsample3d(_ input: GpuTensor, _ base: String, outChannels: Int, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        let cacheName = "\(base).time_conv"
        let valid = validCaches.contains(cacheName)
        var x = input
        if valid {
            x = try causalConv(input, cacheName, outChannels: input.shape.c * 2, kernel: (3, 1, 1), pad: (2, 0, 0), commandBuffer: commandBuffer)
            x = try ops.splitChannelToTime2(x, commandBuffer: commandBuffer)
        } else {
            let cache = try cacheTensor(cacheName, like: input)
            ops.zero(cache, commandBuffer: commandBuffer)
            validCaches.insert(cacheName)
        }
        return try upsample2d(x, base, outChannels: outChannels, commandBuffer: commandBuffer)
    }

    private func normSilu(_ input: GpuTensor, _ base: String, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        let (bias, hasBias) = weights.bias("\(base).bias")
        return try ops.rmsNormSilu(
            input,
            gamma: weightsBuffer("\(base).gamma"),
            bias: bias,
            hasBias: hasBias,
            applySilu: true,
            commandBuffer: commandBuffer
        )
    }

    private func causalConv(
        _ input: GpuTensor,
        _ base: String,
        outChannels: Int,
        kernel: (Int, Int, Int),
        pad: (Int, Int, Int),
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        let cache = try cacheTensor(base, like: input)
        let valid = validCaches.contains(base)
        let out = try conv(input, base, cache: cache, cacheValid: valid, outChannels: outChannels, kernel: kernel, pad: pad, commandBuffer: commandBuffer)
        try ops.updateCache(input: input, cache: cache, hadPrevious: valid, commandBuffer: commandBuffer)
        validCaches.insert(base)
        return out
    }

    private func convNoCache(
        _ input: GpuTensor,
        _ base: String,
        outChannels: Int,
        kernel: (Int, Int, Int),
        pad: (Int, Int, Int),
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        try conv(input, base, cache: nil, cacheValid: false, outChannels: outChannels, kernel: kernel, pad: pad, commandBuffer: commandBuffer)
    }

    private func conv(
        _ input: GpuTensor,
        _ base: String,
        cache: GpuTensor?,
        cacheValid: Bool,
        outChannels: Int,
        kernel: (Int, Int, Int),
        pad: (Int, Int, Int),
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        let (bias, hasBias) = weights.bias("\(base).bias")
        return try ops.conv3d(
            input,
            cache: cache,
            cacheValid: cacheValid,
            weight: weightsBuffer("\(base).kernel"),
            bias: bias,
            hasBias: hasBias,
            outChannels: outChannels,
            kernel: kernel,
            pad: pad,
            commandBuffer: commandBuffer
        )
    }

    private func cacheTensor(_ name: String, like input: GpuTensor) throws -> GpuTensor {
        if let existing = caches[name] {
            return existing
        }
        let cache = try GpuTensor(device: context.device, shape: TensorShape(input.shape.b, 2, input.shape.h, input.shape.w, input.shape.c))
        caches[name] = cache
        return cache
    }

    private func weightsBuffer(_ name: String) throws -> MTLBuffer {
        try weights.buffer(name)
    }
}
