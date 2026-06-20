import Foundation
import Metal

final class MetalOps {
    let context: MetalContext
    private let mpsGraphOps: MPSGraphOps?

    init(context: MetalContext) {
        self.context = context
        if ProcessInfo.processInfo.environment["SOLARIS_DISABLE_MPSGRAPH"] == "1" {
            self.mpsGraphOps = nil
        } else {
            self.mpsGraphOps = MPSGraphOps()
        }
    }

    func scaleLatent(_ input: GpuTensor, mean: MTLBuffer, std: MTLBuffer, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        let out = try GpuTensor(device: context.device, shape: input.shape)
        var params = input.shape.tensorParams
        try encode1D(
            name: "scale_latent4",
            count: (input.shape.elementCount + 3) / 4,
            commandBuffer: commandBuffer
        ) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(mean, offset: 0, index: 1)
            encoder.setBuffer(std, offset: 0, index: 2)
            encoder.setBuffer(out.buffer, offset: 0, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<TensorParams>.stride, index: 4)
        }
        return out
    }

    func conv3d(
        _ input: GpuTensor,
        cache: GpuTensor?,
        cacheValid: Bool,
        weight: MTLBuffer,
        bias: MTLBuffer,
        hasBias: Bool,
        outChannels: Int,
        kernel: (Int, Int, Int),
        pad: (Int, Int, Int),
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        let shape = input.shape
        let outShape = TensorShape(shape.b, shape.t, shape.h, shape.w, outChannels)
        if let mpsGraphOps {
            if let cache, pad.0 > 0 {
                let source = try materializeCausalSource(
                    input,
                    cache: cache,
                    cacheValid: cacheValid,
                    preT: pad.0,
                    commandBuffer: commandBuffer
                )
                return try mpsGraphOps.conv3d(
                    source,
                    weight: weight,
                    bias: bias,
                    hasBias: hasBias,
                    outShape: outShape,
                    kernel: kernel,
                    paddingFront: 0,
                    paddingBack: 0,
                    paddingH: pad.1,
                    paddingW: pad.2,
                    commandBuffer: commandBuffer
                )
            }
            return try mpsGraphOps.conv3d(
                input,
                weight: weight,
                bias: bias,
                hasBias: hasBias,
                outShape: outShape,
                kernel: kernel,
                paddingFront: pad.0,
                paddingBack: 0,
                paddingH: pad.1,
                paddingW: pad.2,
                commandBuffer: commandBuffer
            )
        }
        let out = try GpuTensor(device: context.device, shape: outShape)
        var params = Conv3DParams(
            b: UInt32(shape.b), t: UInt32(shape.t), h: UInt32(shape.h), w: UInt32(shape.w),
            cin: UInt32(shape.c), cout: UInt32(outChannels),
            kt: UInt32(kernel.0), kh: UInt32(kernel.1), kw: UInt32(kernel.2),
            padT: Int32(pad.0), padH: Int32(pad.1), padW: Int32(pad.2),
            cacheT: 2, cacheValid: cacheValid ? 1 : 0, hasBias: hasBias ? 1 : 0
        )
        if kernel.0 == 1, kernel.1 == 1, kernel.2 == 1, pad.0 == 0, pad.1 == 0, pad.2 == 0 {
            try encode2D(
                name: "conv1x1_tiled",
                groups: MTLSize(
                    width: (outChannels + 15) / 16,
                    height: (shape.b * shape.t * shape.h * shape.w + 15) / 16,
                    depth: 1
                ),
                threads: MTLSize(width: 16, height: 16, depth: 1),
                commandBuffer: commandBuffer
            ) { encoder in
                encoder.setBuffer(input.buffer, offset: 0, index: 0)
                encoder.setBuffer(weight, offset: 0, index: 1)
                encoder.setBuffer(bias, offset: 0, index: 2)
                encoder.setBuffer(out.buffer, offset: 0, index: 3)
                encoder.setBytes(&params, length: MemoryLayout<Conv3DParams>.stride, index: 4)
            }
        } else {
            try encode2D(
                name: "conv3d_causal_implicit_gemm",
                groups: MTLSize(
                    width: (outChannels + 15) / 16,
                    height: (shape.b * shape.t * shape.h * shape.w + 15) / 16,
                    depth: 1
                ),
                threads: MTLSize(width: 16, height: 16, depth: 1),
                commandBuffer: commandBuffer
            ) { encoder in
                encoder.setBuffer(input.buffer, offset: 0, index: 0)
                encoder.setBuffer(cache?.buffer ?? input.buffer, offset: 0, index: 1)
                encoder.setBuffer(weight, offset: 0, index: 2)
                encoder.setBuffer(bias, offset: 0, index: 3)
                encoder.setBuffer(out.buffer, offset: 0, index: 4)
                encoder.setBytes(&params, length: MemoryLayout<Conv3DParams>.stride, index: 5)
            }
        }
        return out
    }

    func materializeCausalSource(
        _ input: GpuTensor,
        cache: GpuTensor,
        cacheValid: Bool,
        preT: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        precondition(preT >= 0)
        let out = try GpuTensor(device: context.device, shape: TensorShape(input.shape.b, input.shape.t + preT, input.shape.h, input.shape.w, input.shape.c))
        var params = CausalSourceParams(
            b: UInt32(input.shape.b),
            t: UInt32(input.shape.t),
            h: UInt32(input.shape.h),
            w: UInt32(input.shape.w),
            c: UInt32(input.shape.c),
            preT: UInt32(preT),
            cacheT: UInt32(cache.shape.t),
            cacheValid: cacheValid ? 1 : 0
        )
        try encode1D(name: "materialize_causal_source", count: out.shape.elementCount, commandBuffer: commandBuffer) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(cache.buffer, offset: 0, index: 1)
            encoder.setBuffer(out.buffer, offset: 0, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<CausalSourceParams>.stride, index: 3)
        }
        return out
    }

    func updateCache(input: GpuTensor, cache: GpuTensor, hadPrevious: Bool, commandBuffer: MTLCommandBuffer) throws {
        var params = input.shape.tensorParams
        var previous: UInt32 = hadPrevious ? 1 : 0
        try encode1D(name: "update_cache2", count: cache.shape.elementCount, commandBuffer: commandBuffer) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(cache.buffer, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<TensorParams>.stride, index: 2)
            encoder.setBytes(&previous, length: MemoryLayout<UInt32>.stride, index: 3)
        }
    }

    func zero(_ tensor: GpuTensor, commandBuffer: MTLCommandBuffer) {
        let blit = commandBuffer.makeBlitCommandEncoder()
        blit?.fill(buffer: tensor.buffer, range: 0..<tensor.shape.byteCountF16, value: 0)
        blit?.endEncoding()
    }

    func rmsNormSilu(
        _ input: GpuTensor,
        gamma: MTLBuffer,
        bias: MTLBuffer,
        hasBias: Bool,
        applySilu: Bool,
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        let out = try GpuTensor(device: context.device, shape: input.shape)
        var params = NormParams(
            b: UInt32(input.shape.b), t: UInt32(input.shape.t), h: UInt32(input.shape.h),
            w: UInt32(input.shape.w), c: UInt32(input.shape.c),
            applySilu: applySilu ? 1 : 0, hasBias: hasBias ? 1 : 0
        )
        let vectors = input.shape.b * input.shape.t * input.shape.h * input.shape.w
        try encodeThreadgroups1D(
            name: "rmsnorm_silu_tg",
            groups: vectors,
            threads: 256,
            commandBuffer: commandBuffer
        ) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(gamma, offset: 0, index: 1)
            encoder.setBuffer(bias, offset: 0, index: 2)
            encoder.setBuffer(out.buffer, offset: 0, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<NormParams>.stride, index: 4)
        }
        return out
    }

    func add(_ a: GpuTensor, _ b: GpuTensor, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        precondition(a.shape == b.shape)
        let out = try GpuTensor(device: context.device, shape: a.shape)
        var count = UInt32(a.shape.elementCount)
        try encode1D(name: "add_tensors4", count: (a.shape.elementCount + 3) / 4, commandBuffer: commandBuffer) { encoder in
            encoder.setBuffer(a.buffer, offset: 0, index: 0)
            encoder.setBuffer(b.buffer, offset: 0, index: 1)
            encoder.setBuffer(out.buffer, offset: 0, index: 2)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)
        }
        return out
    }

    func upsample2dConv(
        _ input: GpuTensor,
        weight: MTLBuffer,
        bias: MTLBuffer,
        hasBias: Bool,
        outChannels: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        let shape = input.shape
        if let mpsGraphOps {
            return try mpsGraphOps.upsample2dConv(
                input,
                weight: weight,
                bias: bias,
                hasBias: hasBias,
                outChannels: outChannels,
                commandBuffer: commandBuffer
            )
        }
        let out = try GpuTensor(device: context.device, shape: TensorShape(shape.b, shape.t, shape.h * 2, shape.w * 2, outChannels))
        var params = Up2DConvParams(
            b: UInt32(shape.b), t: UInt32(shape.t), h: UInt32(shape.h), w: UInt32(shape.w),
            cin: UInt32(shape.c), cout: UInt32(outChannels),
            kh: 3, kw: 3, padH: 1, padW: 1, hasBias: hasBias ? 1 : 0
        )
        try encode2D(
            name: "upsample2d_conv2d_implicit_gemm",
            groups: MTLSize(
                width: (outChannels + 15) / 16,
                height: (shape.b * shape.t * shape.h * 2 * shape.w * 2 + 15) / 16,
                depth: 1
            ),
            threads: MTLSize(width: 16, height: 16, depth: 1),
            commandBuffer: commandBuffer
        ) { encoder in
                encoder.setBuffer(input.buffer, offset: 0, index: 0)
                encoder.setBuffer(weight, offset: 0, index: 1)
                encoder.setBuffer(bias, offset: 0, index: 2)
                encoder.setBuffer(out.buffer, offset: 0, index: 3)
                encoder.setBytes(&params, length: MemoryLayout<Up2DConvParams>.stride, index: 4)
        }
        return out
    }

    func phaseUpsample2dConv(
        _ input: GpuTensor,
        foldedWeight: MTLBuffer,
        bias: MTLBuffer,
        hasBias: Bool,
        outChannels: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        guard let mpsGraphOps else {
            throw KVCraftMetalError.invalidArgument("phase upsample requires MPSGraph")
        }
        let phases = try mpsGraphOps.phaseUpsample2dConv(
            input,
            foldedWeight: foldedWeight,
            bias: bias,
            hasBias: hasBias,
            outChannels: outChannels,
            commandBuffer: commandBuffer
        )
        let out = try GpuTensor(device: context.device, shape: TensorShape(input.shape.b, input.shape.t, input.shape.h * 2, input.shape.w * 2, outChannels))
        var params = out.shape.tensorParams
        params.h = UInt32(input.shape.h)
        params.w = UInt32(input.shape.w)
        try encode1D(name: "depth_to_space2_phase_major", count: out.shape.elementCount, commandBuffer: commandBuffer) { encoder in
            encoder.setBuffer(phases.buffer, offset: 0, index: 0)
            encoder.setBuffer(out.buffer, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<TensorParams>.stride, index: 2)
        }
        return out
    }

    func splitChannelToTime2(_ input: GpuTensor, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        precondition(input.shape.c % 2 == 0)
        let out = try GpuTensor(device: context.device, shape: TensorShape(input.shape.b, input.shape.t * 2, input.shape.h, input.shape.w, input.shape.c / 2))
        var params = input.shape.tensorParams
        try encode1D(name: "split_channel_to_time2", count: out.shape.elementCount, commandBuffer: commandBuffer) { encoder in
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(out.buffer, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<TensorParams>.stride, index: 2)
        }
        return out
    }

    func attentionTiled(_ qkv: GpuTensor, channels: Int, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        precondition(qkv.shape.c == channels * 3)
        if let mpsGraphOps {
            let (q, k, v) = try splitQKVForSDPA(qkv, channels: channels, commandBuffer: commandBuffer)
            return try mpsGraphOps.attentionSDPA(
                q: q,
                k: k,
                v: v,
                outputShape: TensorShape(qkv.shape.b, qkv.shape.t, qkv.shape.h, qkv.shape.w, channels),
                channels: channels,
                commandBuffer: commandBuffer
            )
        }
        let out = try GpuTensor(device: context.device, shape: TensorShape(qkv.shape.b, qkv.shape.t, qkv.shape.h, qkv.shape.w, channels))
        var params = TensorParams(
            b: UInt32(qkv.shape.b), t: UInt32(qkv.shape.t), h: UInt32(qkv.shape.h),
            w: UInt32(qkv.shape.w), c: UInt32(channels)
        )
        if channels <= 384 {
            let n = qkv.shape.h * qkv.shape.w
            try encode2D(
                name: "attention_spatial_tiled_384",
                groups: MTLSize(width: (n + 3) / 4, height: qkv.shape.b * qkv.shape.t, depth: 1),
                threads: MTLSize(width: 256, height: 1, depth: 1),
                commandBuffer: commandBuffer
            ) { encoder in
                encoder.setBuffer(qkv.buffer, offset: 0, index: 0)
                encoder.setBuffer(out.buffer, offset: 0, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<TensorParams>.stride, index: 2)
            }
        } else {
            try encode1D(name: "attention_spatial_naive", count: out.shape.elementCount, commandBuffer: commandBuffer) { encoder in
                encoder.setBuffer(qkv.buffer, offset: 0, index: 0)
                encoder.setBuffer(out.buffer, offset: 0, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<TensorParams>.stride, index: 2)
            }
        }
        return out
    }

    private func splitQKVForSDPA(_ qkv: GpuTensor, channels: Int, commandBuffer: MTLCommandBuffer) throws -> (GpuTensor, GpuTensor, GpuTensor) {
        let bt = qkv.shape.b * qkv.shape.t
        let n = qkv.shape.h * qkv.shape.w
        let packedShape = TensorShape(bt, 1, 1, n, channels)
        let q = try GpuTensor(device: context.device, shape: packedShape)
        let k = try GpuTensor(device: context.device, shape: packedShape)
        let v = try GpuTensor(device: context.device, shape: packedShape)
        var params = AttentionSplitParams(bt: UInt32(bt), n: UInt32(n), c: UInt32(channels))
        try encode1D(name: "split_qkv_for_sdpa", count: packedShape.elementCount, commandBuffer: commandBuffer) { encoder in
            encoder.setBuffer(qkv.buffer, offset: 0, index: 0)
            encoder.setBuffer(q.buffer, offset: 0, index: 1)
            encoder.setBuffer(k.buffer, offset: 0, index: 2)
            encoder.setBuffer(v.buffer, offset: 0, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<AttentionSplitParams>.stride, index: 4)
        }
        return (q, k, v)
    }

    func attentionNaive(_ qkv: GpuTensor, channels: Int, commandBuffer: MTLCommandBuffer) throws -> GpuTensor {
        try attentionTiled(qkv, channels: channels, commandBuffer: commandBuffer)
    }

    private func encode1D(
        name: String,
        count: Int,
        commandBuffer: MTLCommandBuffer,
        bind: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try context.pipeline(name)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KVCraftMetalError.allocationFailed("compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        bind(encoder)
        let width = min(max(1, pipeline.threadExecutionWidth), 256)
        let threads = MTLSize(width: width, height: 1, depth: 1)
        let groups = MTLSize(width: (count + width - 1) / width, height: 1, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }

    private func encodeThreadgroups1D(
        name: String,
        groups: Int,
        threads: Int,
        commandBuffer: MTLCommandBuffer,
        bind: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try context.pipeline(name)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KVCraftMetalError.allocationFailed("compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        bind(encoder)
        encoder.dispatchThreadgroups(
            MTLSize(width: max(1, groups), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        encoder.endEncoding()
    }

    private func encode2D(
        name: String,
        groups: MTLSize,
        threads: MTLSize,
        commandBuffer: MTLCommandBuffer,
        bind: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try context.pipeline(name)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw KVCraftMetalError.allocationFailed("compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        bind(encoder)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        encoder.endEncoding()
    }
}
