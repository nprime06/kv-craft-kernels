import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

final class SteadyStateGraphExecutor {
    static let requiredCacheNames: [String] = {
        var names = ["decoder.conv1"]
        for block in residualBlockNames {
            names.append("\(block).residual.2")
            names.append("\(block).residual.6")
        }
        names.append("decoder.upsamples.7.time_conv")
        names.append("decoder.upsamples.11.time_conv")
        names.append("decoder.head.2")
        return names
    }()

    private let context: MetalContext
    private let weights: WeightArchive
    private let latentHeight: Int
    private let latentWidth: Int
    private let skipAttention: Bool
    private var plan: SteadyStatePlan?

    init(context: MetalContext, weights: WeightArchive, latentHeight: Int, latentWidth: Int, skipAttention: Bool) {
        self.context = context
        self.weights = weights
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.skipAttention = skipAttention
    }

    func canRun(validCaches: Set<String>) -> Bool {
        Self.requiredCacheNames.allSatisfy { validCaches.contains($0) }
    }

    func encode(
        latent: GpuTensor,
        caches: [String: GpuTensor],
        commandBuffer: MTLCommandBuffer
    ) throws -> (output: GpuTensor, updatedCaches: [String: GpuTensor], retainedInputs: [GpuTensor]) {
        let plan = try steadyPlan()
        let output = try GpuTensor(device: context.device, shape: plan.outputShape)

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.latent: MPSGraphTensorData(latent.buffer, shape: shape5(latent.shape), dataType: .float16)
        ]
        for feed in plan.weightFeeds {
            feeds[feed.tensor] = MPSGraphTensorData(try weights.buffer(feed.name), shape: feed.shape, dataType: .float16)
        }
        for feed in plan.cacheFeeds {
            guard let cache = caches[feed.name] else {
                throw KVCraftMetalError.invalidArgument("missing steady-state cache \(feed.name)")
            }
            feeds[feed.tensor] = MPSGraphTensorData(cache.buffer, shape: shape5(feed.shape), dataType: .float16)
        }

        var results: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.output: MPSGraphTensorData(output.buffer, shape: shape5(plan.outputShape), dataType: .float16)
        ]
        var updated: [String: GpuTensor] = [:]
        updated.reserveCapacity(plan.cacheOutputs.count)
        for cacheOutput in plan.cacheOutputs {
            let tensor = try GpuTensor(device: context.device, shape: cacheOutput.shape)
            updated[cacheOutput.name] = tensor
            results[cacheOutput.tensor] = MPSGraphTensorData(tensor.buffer, shape: shape5(cacheOutput.shape), dataType: .float16)
        }

        plan.graph.encode(
            to: mpsCommandBuffer(commandBuffer),
            feeds: feeds,
            targetOperations: nil,
            resultsDictionary: results,
            executionDescriptor: nil
        )

        let retainedInputs = Array(caches.values) + [latent]
        return (output, updated, retainedInputs)
    }

    private func steadyPlan() throws -> SteadyStatePlan {
        if let plan {
            return plan
        }
        let builder = try SteadyStateGraphBuilder(
            weights: weights,
            latentHeight: latentHeight,
            latentWidth: latentWidth,
            skipAttention: skipAttention
        )
        let built = try builder.build()
        plan = built
        return built
    }

    private func mpsCommandBuffer(_ commandBuffer: MTLCommandBuffer) -> MPSCommandBuffer {
        if let existing = commandBuffer as? MPSCommandBuffer {
            return existing
        }
        return MPSCommandBuffer(commandBuffer: commandBuffer)
    }
}

private final class SteadyStateGraphBuilder {
    private let weights: WeightArchive
    private let latentHeight: Int
    private let latentWidth: Int
    private let skipAttention: Bool
    private let useNativeConv3D: Bool
    private let useF16Norm: Bool
    private let graph = MPSGraph()

    private var weightFeeds: [GraphFeed] = []
    private var cacheFeeds: [CacheFeed] = []
    private var cacheTensors: [String: GraphValue] = [:]
    private var cacheOutputs: [CacheOutput] = []

    init(weights: WeightArchive, latentHeight: Int, latentWidth: Int, skipAttention: Bool) throws {
        self.weights = weights
        self.latentHeight = latentHeight
        self.latentWidth = latentWidth
        self.skipAttention = skipAttention
        self.useNativeConv3D = ProcessInfo.processInfo.environment["SOLARIS_STEADY_NATIVE_CONV3D"] == "1"
        self.useF16Norm = ProcessInfo.processInfo.environment["SOLARIS_STEADY_F16_NORM"] == "1"
    }

    func build() throws -> SteadyStatePlan {
        let latentShape = TensorShape(1, 1, latentHeight, latentWidth, 16)
        let latent = graph.placeholder(shape: shape5(latentShape), dataType: .float16, name: "latent")
        var x = GraphValue(tensor: latent, shape: latentShape)
        x = try scaleLatent(x)
        x = try convNoCache(x, "conv2", outChannels: 16, kernel: (1, 1, 1), pad: (0, 0, 0))
        let output = try decodeCore(x)

        return SteadyStatePlan(
            graph: graph,
            latent: latent,
            output: output.tensor,
            outputShape: output.shape,
            weightFeeds: weightFeeds,
            cacheFeeds: cacheFeeds,
            cacheOutputs: cacheOutputs
        )
    }

    private func decodeCore(_ input: GraphValue) throws -> GraphValue {
        var x = try causalConv(input, "decoder.conv1", outChannels: 384, kernel: (3, 3, 3), pad: (2, 1, 1))

        x = try residual(x, "decoder.middle.0", outChannels: 384)
        x = try attention(x, "decoder.middle.1", channels: 384)
        x = try residual(x, "decoder.middle.2", outChannels: 384)

        x = try residual(x, "decoder.upsamples.0", outChannels: 384)
        x = try residual(x, "decoder.upsamples.1", outChannels: 384)
        x = try residual(x, "decoder.upsamples.2", outChannels: 384)
        x = try upsample2d(x, "decoder.upsamples.3", outChannels: 192)

        x = try residual(x, "decoder.upsamples.4", outChannels: 384)
        x = try residual(x, "decoder.upsamples.5", outChannels: 384)
        x = try residual(x, "decoder.upsamples.6", outChannels: 384)
        x = try upsample3d(x, "decoder.upsamples.7", outChannels: 192)

        x = try residual(x, "decoder.upsamples.8", outChannels: 192)
        x = try residual(x, "decoder.upsamples.9", outChannels: 192)
        x = try residual(x, "decoder.upsamples.10", outChannels: 192)
        x = try upsample3d(x, "decoder.upsamples.11", outChannels: 96)

        x = try residual(x, "decoder.upsamples.12", outChannels: 96)
        x = try residual(x, "decoder.upsamples.13", outChannels: 96)
        x = try residual(x, "decoder.upsamples.14", outChannels: 96)

        x = try normSilu(x, "decoder.head.0", applySilu: true)
        x = try causalConv(x, "decoder.head.2", outChannels: 3, kernel: (3, 3, 3), pad: (2, 1, 1))
        return x
    }

    private func residual(_ input: GraphValue, _ base: String, outChannels: Int) throws -> GraphValue {
        let shortcut: GraphValue
        if input.shape.c == outChannels {
            shortcut = input
        } else {
            shortcut = try convNoCache(input, "\(base).shortcut", outChannels: outChannels, kernel: (1, 1, 1), pad: (0, 0, 0))
        }

        var x = try normSilu(input, "\(base).residual.0", applySilu: true)
        x = try causalConv(x, "\(base).residual.2", outChannels: outChannels, kernel: (3, 3, 3), pad: (2, 1, 1))
        x = try normSilu(x, "\(base).residual.3", applySilu: true)
        x = try causalConv(x, "\(base).residual.6", outChannels: outChannels, kernel: (3, 3, 3), pad: (2, 1, 1))
        return GraphValue(tensor: graph.addition(x.tensor, shortcut.tensor, name: "\(base).add"), shape: x.shape)
    }

    private func attention(_ input: GraphValue, _ base: String, channels: Int) throws -> GraphValue {
        if skipAttention {
            return input
        }
        let normed = try normSilu(input, "\(base).norm", applySilu: false)
        let qkv = try convNoCache(normed, "\(base).to_qkv", outChannels: channels * 3, kernel: (1, 1, 1), pad: (0, 0, 0))

        let bt = qkv.shape.b * qkv.shape.t
        let n = qkv.shape.h * qkv.shape.w
        let packed = graph.reshape(qkv.tensor, shape: nums([bt, n, channels * 3]), name: "\(base).qkv_3d")
        let q = graph.reshape(
            graph.sliceTensor(packed, dimension: 2, start: 0, length: channels, name: "\(base).q"),
            shape: nums([bt, 1, n, channels]),
            name: "\(base).q4"
        )
        let k = graph.reshape(
            graph.sliceTensor(packed, dimension: 2, start: channels, length: channels, name: "\(base).k"),
            shape: nums([bt, 1, n, channels]),
            name: "\(base).k4"
        )
        let v = graph.reshape(
            graph.sliceTensor(packed, dimension: 2, start: channels * 2, length: channels, name: "\(base).v"),
            shape: nums([bt, 1, n, channels]),
            name: "\(base).v4"
        )
        let scale = Float(1.0 / sqrt(Double(channels)))
        let attended = graph.scaledDotProductAttention(query: q, key: k, value: v, scale: scale, name: "\(base).sdpa")
        let attended5 = GraphValue(
            tensor: graph.reshape(attended, shape: shape5(input.shape), name: "\(base).attended5"),
            shape: input.shape
        )
        let projected = try convNoCache(attended5, "\(base).proj", outChannels: channels, kernel: (1, 1, 1), pad: (0, 0, 0))
        return GraphValue(tensor: graph.addition(projected.tensor, input.tensor, name: "\(base).add"), shape: input.shape)
    }

    private func upsample3d(_ input: GraphValue, _ base: String, outChannels: Int) throws -> GraphValue {
        var x = try causalConv(input, "\(base).time_conv", outChannels: input.shape.c * 2, kernel: (3, 1, 1), pad: (2, 0, 0))
        x = splitChannelToTime2(x, name: "\(base).time_split")
        return try upsample2d(x, base, outChannels: outChannels)
    }

    private func scaleLatent(_ input: GraphValue) throws -> GraphValue {
        let mean = weight("vae_scale.mean", shape: nums([1, 1, 1, 1, input.shape.c]))
        let std = weight("vae_scale.std", shape: nums([1, 1, 1, 1, input.shape.c]))
        let scaled = graph.multiplication(input.tensor, std, name: "latent_scale_std")
        return GraphValue(tensor: graph.addition(scaled, mean, name: "latent_scale_mean"), shape: input.shape)
    }

    private func normSilu(_ input: GraphValue, _ base: String, applySilu: Bool) throws -> GraphValue {
        let gamma = weight("\(base).gamma", shape: nums([1, 1, 1, 1, input.shape.c]))
        if useF16Norm {
            return normSiluF16(input, base, gamma: gamma, applySilu: applySilu)
        }
        let x32 = graph.cast(input.tensor, to: .float32, name: "\(base).x32")
        let square = graph.square(with: x32, name: "\(base).square")
        let sum = graph.reductionSum(with: square, axes: [4], name: "\(base).sum")
        let sum5 = graph.reshape(sum, shape: nums([input.shape.b, input.shape.t, input.shape.h, input.shape.w, 1]), name: "\(base).sum5")
        let epsilon = graph.constant(1.0e-12, dataType: .float32)
        let safeSum = graph.maximum(sum5, epsilon, name: "\(base).safe_sum")
        let inv = graph.reciprocalSquareRoot(safeSum, name: "\(base).rsqrt")
        let scale = graph.constant(sqrt(Double(input.shape.c)), dataType: .float32)
        var y = graph.multiplication(x32, inv, name: "\(base).norm_mul_inv")
        y = graph.multiplication(y, scale, name: "\(base).norm_scale")
        y = graph.multiplication(y, graph.cast(gamma, to: .float32, name: "\(base).gamma32"), name: "\(base).norm_gamma")
        let (biasBuffer, hasBias) = weights.bias("\(base).bias")
        _ = biasBuffer
        if hasBias {
            let bias = weight("\(base).bias", shape: nums([1, 1, 1, 1, input.shape.c]))
            y = graph.addition(y, graph.cast(bias, to: .float32, name: "\(base).bias32"), name: "\(base).bias_add")
        }
        if applySilu {
            y = graph.multiplication(y, graph.sigmoid(with: y, name: "\(base).sigmoid"), name: "\(base).silu")
        }
        return GraphValue(tensor: graph.cast(y, to: .float16, name: "\(base).f16"), shape: input.shape)
    }

    private func normSiluF16(_ input: GraphValue, _ base: String, gamma: MPSGraphTensor, applySilu: Bool) -> GraphValue {
        let square = graph.square(with: input.tensor, name: "\(base).square_f16")
        let sum = graph.reductionSum(with: square, axes: [4], name: "\(base).sum_f16")
        let sum5 = graph.reshape(sum, shape: nums([input.shape.b, input.shape.t, input.shape.h, input.shape.w, 1]), name: "\(base).sum5_f16")
        let epsilon = graph.constant(1.0e-7, dataType: .float16)
        let safeSum = graph.maximum(sum5, epsilon, name: "\(base).safe_sum_f16")
        let inv = graph.reciprocalSquareRoot(safeSum, name: "\(base).rsqrt_f16")
        let scale = graph.constant(sqrt(Double(input.shape.c)), dataType: .float16)
        var y = graph.multiplication(input.tensor, inv, name: "\(base).norm_mul_inv_f16")
        y = graph.multiplication(y, scale, name: "\(base).norm_scale_f16")
        y = graph.multiplication(y, gamma, name: "\(base).norm_gamma_f16")
        let (biasBuffer, hasBias) = weights.bias("\(base).bias")
        _ = biasBuffer
        if hasBias {
            let bias = weight("\(base).bias", shape: nums([1, 1, 1, 1, input.shape.c]))
            y = graph.addition(y, bias, name: "\(base).bias_add_f16")
        }
        if applySilu {
            y = graph.multiplication(y, graph.sigmoid(with: y, name: "\(base).sigmoid_f16"), name: "\(base).silu_f16")
        }
        return GraphValue(tensor: y, shape: input.shape)
    }

    private func convNoCache(
        _ input: GraphValue,
        _ base: String,
        outChannels: Int,
        kernel: (Int, Int, Int),
        pad: (Int, Int, Int)
    ) throws -> GraphValue {
        try convTemporal(input, base, source: input, outChannels: outChannels, kernel: kernel, padH: pad.1, padW: pad.2)
    }

    private func causalConv(
        _ input: GraphValue,
        _ base: String,
        outChannels: Int,
        kernel: (Int, Int, Int),
        pad: (Int, Int, Int)
    ) throws -> GraphValue {
        let cache = cachePlaceholder(base, like: input)
        let source = GraphValue(
            tensor: graph.concatTensors([cache.tensor, input.tensor], dimension: 1, name: "\(base).source"),
            shape: TensorShape(input.shape.b, input.shape.t + 2, input.shape.h, input.shape.w, input.shape.c)
        )
        let out = try convTemporal(input, base, source: source, outChannels: outChannels, kernel: kernel, padH: pad.1, padW: pad.2)
        let updated = updateCache(input: input, oldCache: cache, name: base)
        cacheOutputs.append(CacheOutput(name: base, tensor: updated.tensor, shape: updated.shape))
        return out
    }

    private func convTemporal(
        _ input: GraphValue,
        _ base: String,
        source: GraphValue,
        outChannels: Int,
        kernel: (Int, Int, Int),
        padH: Int,
        padW: Int
    ) throws -> GraphValue {
        let outShape = TensorShape(input.shape.b, input.shape.t, input.shape.h, input.shape.w, outChannels)
        guard source.shape.t == outShape.t + kernel.0 - 1 else {
            throw KVCraftMetalError.invalidArgument("steady graph conv \(base) has incompatible time shape")
        }
        let weightTensor = weight("\(base).kernel", shape: nums([kernel.0, kernel.1, kernel.2, input.shape.c, outChannels]))
        if useNativeConv3D {
            return try convNative3D(
                source: source,
                base: base,
                outShape: outShape,
                outChannels: outChannels,
                kernel: kernel,
                padH: padH,
                padW: padW,
                weightTensor: weightTensor
            )
        }
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: padW,
            paddingRight: padW,
            paddingTop: padH,
            paddingBottom: padH,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        ) else {
            throw KVCraftMetalError.allocationFailed("steady graph convolution descriptor")
        }

        var sum: MPSGraphTensor?
        for kz in 0..<kernel.0 {
            let inputSlice = graph.sliceTensor(source.tensor, dimension: 1, start: kz, length: outShape.t, name: "\(base).input_t\(kz)")
            let input2D = graph.reshape(
                inputSlice,
                shape: nums([outShape.b * outShape.t, outShape.h, outShape.w, input.shape.c]),
                name: "\(base).input2d_t\(kz)"
            )
            let weightSlice = graph.sliceTensor(weightTensor, dimension: 0, start: kz, length: 1, name: "\(base).weight_t\(kz)")
            let weight2D = graph.reshape(
                weightSlice,
                shape: nums([kernel.1, kernel.2, input.shape.c, outChannels]),
                name: "\(base).weight2d_t\(kz)"
            )
            let conv = graph.convolution2D(input2D, weights: weight2D, descriptor: desc, name: "\(base).conv2d_t\(kz)")
            if let existing = sum {
                sum = graph.addition(existing, conv, name: "\(base).sum_t\(kz)")
            } else {
                sum = conv
            }
        }

        guard let output2D = sum else {
            throw KVCraftMetalError.invalidArgument("empty steady graph convolution \(base)")
        }
        var output = graph.reshape(output2D, shape: shape5(outShape), name: "\(base).output5")
        let (biasBuffer, hasBias) = weights.bias("\(base).bias")
        _ = biasBuffer
        if hasBias {
            let bias = weight("\(base).bias", shape: nums([1, 1, 1, 1, outChannels]))
            output = graph.addition(output, bias, name: "\(base).bias_add")
        }
        return GraphValue(tensor: output, shape: outShape)
    }

    private func convNative3D(
        source: GraphValue,
        base: String,
        outShape: TensorShape,
        outChannels: Int,
        kernel: (Int, Int, Int),
        padH: Int,
        padW: Int,
        weightTensor: MPSGraphTensor
    ) throws -> GraphValue {
        guard let desc = MPSGraphConvolution3DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            strideInZ: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            dilationRateInZ: 1,
            groups: 1,
            paddingLeft: padW,
            paddingRight: padW,
            paddingTop: padH,
            paddingBottom: padH,
            paddingFront: 0,
            paddingBack: 0,
            paddingStyle: .explicit,
            dataLayout: .NDHWC,
            weightsLayout: .DHWIO
        ) else {
            throw KVCraftMetalError.allocationFailed("steady graph native 3D convolution descriptor")
        }
        var output = graph.convolution3D(source.tensor, weights: weightTensor, descriptor: desc, name: "\(base).conv3d")
        let (biasBuffer, hasBias) = weights.bias("\(base).bias")
        _ = biasBuffer
        if hasBias {
            let bias = weight("\(base).bias", shape: nums([1, 1, 1, 1, outChannels]))
            output = graph.addition(output, bias, name: "\(base).bias_add")
        }
        return GraphValue(tensor: output, shape: outShape)
    }

    private func upsample2d(_ input: GraphValue, _ base: String, outChannels: Int) throws -> GraphValue {
        let outShape = TensorShape(input.shape.b, input.shape.t, input.shape.h * 2, input.shape.w * 2, outChannels)
        let input2D = graph.reshape(input.tensor, shape: nums([input.shape.b * input.shape.t, input.shape.h, input.shape.w, input.shape.c]), name: "\(base).up_input2d")
        let resizeSize = [Int32(outShape.h), Int32(outShape.w)]
        let resizeData = resizeSize.withUnsafeBufferPointer { Data(buffer: $0) }
        let size = graph.constant(resizeData, shape: nums([2]), dataType: .int32)
        let resized = graph.resizeNearest(
            input2D,
            sizeTensor: size,
            nearestRoundingMode: .floor,
            centerResult: false,
            alignCorners: false,
            layout: .NHWC,
            name: "\(base).nearest2x"
        )
        let weightTensor = weight("\(base).conv.kernel", shape: nums([3, 3, input.shape.c, outChannels]))
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: 1,
            paddingRight: 1,
            paddingTop: 1,
            paddingBottom: 1,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        ) else {
            throw KVCraftMetalError.allocationFailed("steady graph upsample convolution descriptor")
        }
        var output = graph.convolution2D(resized, weights: weightTensor, descriptor: desc, name: "\(base).up_conv")
        let (biasBuffer, hasBias) = weights.bias("\(base).conv.bias")
        _ = biasBuffer
        if hasBias {
            let bias = weight("\(base).conv.bias", shape: nums([1, 1, 1, outChannels]))
            output = graph.addition(output, bias, name: "\(base).up_bias")
        }
        return GraphValue(tensor: graph.reshape(output, shape: shape5(outShape), name: "\(base).up_output5"), shape: outShape)
    }

    private func splitChannelToTime2(_ input: GraphValue, name: String) -> GraphValue {
        let outChannels = input.shape.c / 2
        let reshaped = graph.reshape(
            input.tensor,
            shape: nums([input.shape.b, input.shape.t, input.shape.h, input.shape.w, 2, outChannels]),
            name: "\(name).reshape6"
        )
        let transposed = graph.transpose(reshaped, permutation: nums([0, 1, 4, 2, 3, 5]), name: "\(name).transpose")
        let outShape = TensorShape(input.shape.b, input.shape.t * 2, input.shape.h, input.shape.w, outChannels)
        return GraphValue(tensor: graph.reshape(transposed, shape: shape5(outShape), name: "\(name).output5"), shape: outShape)
    }

    private func updateCache(input: GraphValue, oldCache: GraphValue, name: String) -> GraphValue {
        let outShape = oldCache.shape
        if input.shape.t >= 2 {
            return GraphValue(
                tensor: graph.sliceTensor(input.tensor, dimension: 1, start: input.shape.t - 2, length: 2, name: "\(name).cache_update"),
                shape: outShape
            )
        }
        let previous = graph.sliceTensor(oldCache.tensor, dimension: 1, start: 1, length: 1, name: "\(name).cache_prev")
        return GraphValue(
            tensor: graph.concatTensors([previous, input.tensor], dimension: 1, name: "\(name).cache_update"),
            shape: outShape
        )
    }

    private func cachePlaceholder(_ name: String, like input: GraphValue) -> GraphValue {
        if let existing = cacheTensors[name] {
            return existing
        }
        let shape = TensorShape(input.shape.b, 2, input.shape.h, input.shape.w, input.shape.c)
        let tensor = graph.placeholder(shape: shape5(shape), dataType: .float16, name: "\(name).cache")
        let value = GraphValue(tensor: tensor, shape: shape)
        cacheTensors[name] = value
        cacheFeeds.append(CacheFeed(name: name, tensor: tensor, shape: shape))
        return value
    }

    private func weight(_ name: String, shape: [NSNumber]) -> MPSGraphTensor {
        let tensor = graph.placeholder(shape: shape, dataType: .float16, name: name)
        weightFeeds.append(GraphFeed(name: name, tensor: tensor, shape: shape))
        return tensor
    }
}

private let residualBlockNames = [
    "decoder.middle.0",
    "decoder.middle.2",
    "decoder.upsamples.0",
    "decoder.upsamples.1",
    "decoder.upsamples.2",
    "decoder.upsamples.4",
    "decoder.upsamples.5",
    "decoder.upsamples.6",
    "decoder.upsamples.8",
    "decoder.upsamples.9",
    "decoder.upsamples.10",
    "decoder.upsamples.12",
    "decoder.upsamples.13",
    "decoder.upsamples.14",
]

private struct GraphValue {
    let tensor: MPSGraphTensor
    let shape: TensorShape
}

private struct GraphFeed {
    let name: String
    let tensor: MPSGraphTensor
    let shape: [NSNumber]
}

private struct CacheFeed {
    let name: String
    let tensor: MPSGraphTensor
    let shape: TensorShape
}

private struct CacheOutput {
    let name: String
    let tensor: MPSGraphTensor
    let shape: TensorShape
}

private final class SteadyStatePlan {
    let graph: MPSGraph
    let latent: MPSGraphTensor
    let output: MPSGraphTensor
    let outputShape: TensorShape
    let weightFeeds: [GraphFeed]
    let cacheFeeds: [CacheFeed]
    let cacheOutputs: [CacheOutput]

    init(
        graph: MPSGraph,
        latent: MPSGraphTensor,
        output: MPSGraphTensor,
        outputShape: TensorShape,
        weightFeeds: [GraphFeed],
        cacheFeeds: [CacheFeed],
        cacheOutputs: [CacheOutput]
    ) {
        self.graph = graph
        self.latent = latent
        self.output = output
        self.outputShape = outputShape
        self.weightFeeds = weightFeeds
        self.cacheFeeds = cacheFeeds
        self.cacheOutputs = cacheOutputs
    }
}

private func shape5(_ shape: TensorShape) -> [NSNumber] {
    nums([shape.b, shape.t, shape.h, shape.w, shape.c])
}

private func nums(_ values: [Int]) -> [NSNumber] {
    values.map { NSNumber(value: $0) }
}
