import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

final class MPSGraphOps {
    private var conv3DPlans: [Conv3DKey: Conv3DPlan] = [:]
    private var upsample2DPlans: [Upsample2DKey: Upsample2DPlan] = [:]
    private var phaseUpsample2DPlans: [PhaseUpsample2DKey: PhaseUpsample2DPlan] = [:]
    private var attentionPlans: [AttentionKey: AttentionPlan] = [:]
    private let decomposeConv3DTo2D = ProcessInfo.processInfo.environment["SOLARIS_DISABLE_CONV3D_AS_2D"] != "1"

    func conv3d(
        _ input: GpuTensor,
        weight: MTLBuffer,
        bias: MTLBuffer,
        hasBias: Bool,
        outShape: TensorShape,
        kernel: (Int, Int, Int),
        paddingFront: Int,
        paddingBack: Int,
        paddingH: Int,
        paddingW: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        let key = Conv3DKey(
            inputShape: input.shape,
            outputShape: outShape,
            kt: kernel.0,
            kh: kernel.1,
            kw: kernel.2,
            paddingFront: paddingFront,
            paddingBack: paddingBack,
            paddingH: paddingH,
            paddingW: paddingW,
            hasBias: hasBias
        )
        let plan = try conv3DPlan(key)
        let out = try GpuTensor(device: input.buffer.device, shape: outShape)

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.input: MPSGraphTensorData(input.buffer, shape: shape5(input.shape), dataType: .float16),
            plan.weight: MPSGraphTensorData(
                weight,
                shape: nums([kernel.0, kernel.1, kernel.2, input.shape.c, outShape.c]),
                dataType: .float16
            ),
        ]
        if let biasTensor = plan.bias {
            feeds[biasTensor] = MPSGraphTensorData(
                bias,
                shape: nums([1, 1, 1, 1, outShape.c]),
                dataType: .float16
            )
        }
        let results = [
            plan.output: MPSGraphTensorData(out.buffer, shape: shape5(outShape), dataType: .float16)
        ]
        plan.graph.encode(
            to: mpsCommandBuffer(commandBuffer),
            feeds: feeds,
            targetOperations: nil,
            resultsDictionary: results,
            executionDescriptor: nil
        )
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
        let outShape = TensorShape(input.shape.b, input.shape.t, input.shape.h * 2, input.shape.w * 2, outChannels)
        let key = Upsample2DKey(inputShape: input.shape, outputShape: outShape, hasBias: hasBias)
        let plan = try upsample2DPlan(key)
        let out = try GpuTensor(device: input.buffer.device, shape: outShape)

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.input: MPSGraphTensorData(input.buffer, shape: shape4BT(input.shape), dataType: .float16),
            plan.weight: MPSGraphTensorData(weight, shape: nums([3, 3, input.shape.c, outChannels]), dataType: .float16),
        ]
        if let biasTensor = plan.bias {
            feeds[biasTensor] = MPSGraphTensorData(bias, shape: nums([1, 1, 1, outChannels]), dataType: .float16)
        }
        let results = [
            plan.output: MPSGraphTensorData(out.buffer, shape: shape4BT(outShape), dataType: .float16)
        ]
        plan.graph.encode(
            to: mpsCommandBuffer(commandBuffer),
            feeds: feeds,
            targetOperations: nil,
            resultsDictionary: results,
            executionDescriptor: nil
        )
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
        let phaseShape = TensorShape(input.shape.b, input.shape.t, input.shape.h, input.shape.w, outChannels * 4)
        let key = PhaseUpsample2DKey(inputShape: input.shape, outChannels: outChannels, hasBias: hasBias)
        let plan = try phaseUpsample2DPlan(key)
        let out = try GpuTensor(device: input.buffer.device, shape: phaseShape)

        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.input: MPSGraphTensorData(input.buffer, shape: shape4BT(input.shape), dataType: .float16),
            plan.weight: MPSGraphTensorData(foldedWeight, shape: nums([4, 2, 2, input.shape.c, outChannels]), dataType: .float16),
        ]
        if let biasTensor = plan.bias {
            feeds[biasTensor] = MPSGraphTensorData(bias, shape: nums([1, 1, 1, outChannels]), dataType: .float16)
        }
        let results = [
            plan.output: MPSGraphTensorData(out.buffer, shape: shape4BT(phaseShape), dataType: .float16)
        ]
        plan.graph.encode(
            to: mpsCommandBuffer(commandBuffer),
            feeds: feeds,
            targetOperations: nil,
            resultsDictionary: results,
            executionDescriptor: nil
        )
        return out
    }

    func attentionSDPA(
        q: GpuTensor,
        k: GpuTensor,
        v: GpuTensor,
        outputShape: TensorShape,
        channels: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> GpuTensor {
        let bt = outputShape.b * outputShape.t
        let n = outputShape.h * outputShape.w
        let key = AttentionKey(bt: bt, n: n, c: channels)
        let plan = attentionPlan(key)
        let out = try GpuTensor(device: q.buffer.device, shape: outputShape)

        let sdpaShape = nums([bt, 1, n, channels])
        let feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            plan.q: MPSGraphTensorData(q.buffer, shape: sdpaShape, dataType: .float16),
            plan.k: MPSGraphTensorData(k.buffer, shape: sdpaShape, dataType: .float16),
            plan.v: MPSGraphTensorData(v.buffer, shape: sdpaShape, dataType: .float16),
        ]
        let results = [
            plan.output: MPSGraphTensorData(out.buffer, shape: sdpaShape, dataType: .float16)
        ]
        plan.graph.encode(
            to: mpsCommandBuffer(commandBuffer),
            feeds: feeds,
            targetOperations: nil,
            resultsDictionary: results,
            executionDescriptor: nil
        )
        return out
    }

    private func conv3DPlan(_ key: Conv3DKey) throws -> Conv3DPlan {
        if let existing = conv3DPlans[key] {
            return existing
        }
        if decomposeConv3DTo2D, canUse2DConvDecomposition(key) {
            let plan = try conv3DAs2DPlan(key)
            conv3DPlans[key] = plan
            return plan
        }
        let graph = MPSGraph()
        let input = graph.placeholder(shape: shape5(key.inputShape), dataType: .float16, name: "input")
        let weight = graph.placeholder(
            shape: nums([key.kt, key.kh, key.kw, key.inputShape.c, key.outputShape.c]),
            dataType: .float16,
            name: "weight"
        )
        guard let desc = MPSGraphConvolution3DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            strideInZ: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            dilationRateInZ: 1,
            groups: 1,
            paddingLeft: key.paddingW,
            paddingRight: key.paddingW,
            paddingTop: key.paddingH,
            paddingBottom: key.paddingH,
            paddingFront: key.paddingFront,
            paddingBack: key.paddingBack,
            paddingStyle: .explicit,
            dataLayout: .NDHWC,
            weightsLayout: .DHWIO
        ) else {
            throw KVCraftMetalError.allocationFailed("MPSGraph 3D convolution descriptor")
        }
        var output = graph.convolution3D(input, weights: weight, descriptor: desc, name: "conv3d")
        var biasTensor: MPSGraphTensor?
        if key.hasBias {
            let bias = graph.placeholder(shape: nums([1, 1, 1, 1, key.outputShape.c]), dataType: .float16, name: "bias")
            output = graph.addition(output, bias, name: "bias_add")
            biasTensor = bias
        }
        let plan = Conv3DPlan(graph: graph, input: input, weight: weight, bias: biasTensor, output: output)
        conv3DPlans[key] = plan
        return plan
    }

    private func canUse2DConvDecomposition(_ key: Conv3DKey) -> Bool {
        key.paddingFront == 0 &&
            key.paddingBack == 0 &&
            key.inputShape.t == key.outputShape.t + key.kt - 1
    }

    private func conv3DAs2DPlan(_ key: Conv3DKey) throws -> Conv3DPlan {
        let graph = MPSGraph()
        let input = graph.placeholder(shape: shape5(key.inputShape), dataType: .float16, name: "input")
        let weight = graph.placeholder(
            shape: nums([key.kt, key.kh, key.kw, key.inputShape.c, key.outputShape.c]),
            dataType: .float16,
            name: "weight"
        )
        guard let desc = MPSGraphConvolution2DOpDescriptor(
            strideInX: 1,
            strideInY: 1,
            dilationRateInX: 1,
            dilationRateInY: 1,
            groups: 1,
            paddingLeft: key.paddingW,
            paddingRight: key.paddingW,
            paddingTop: key.paddingH,
            paddingBottom: key.paddingH,
            paddingStyle: .explicit,
            dataLayout: .NHWC,
            weightsLayout: .HWIO
        ) else {
            throw KVCraftMetalError.allocationFailed("MPSGraph 2D convolution descriptor")
        }

        var sum: MPSGraphTensor?
        for kz in 0..<key.kt {
            let inputSlice = graph.sliceTensor(input, dimension: 1, start: kz, length: key.outputShape.t, name: "input_t\(kz)")
            let input2D = graph.reshape(
                inputSlice,
                shape: nums([key.outputShape.b * key.outputShape.t, key.outputShape.h, key.outputShape.w, key.inputShape.c]),
                name: "input_2d_t\(kz)"
            )
            let weightSlice = graph.sliceTensor(weight, dimension: 0, start: kz, length: 1, name: "weight_t\(kz)")
            let weight2D = graph.reshape(
                weightSlice,
                shape: nums([key.kh, key.kw, key.inputShape.c, key.outputShape.c]),
                name: "weight_2d_t\(kz)"
            )
            let conv = graph.convolution2D(input2D, weights: weight2D, descriptor: desc, name: "conv2d_t\(kz)")
            if let existing = sum {
                sum = graph.addition(existing, conv, name: "sum_t\(kz)")
            } else {
                sum = conv
            }
        }

        guard let output2D = sum else {
            throw KVCraftMetalError.invalidArgument("empty convolution kernel")
        }
        var output = graph.reshape(output2D, shape: shape5(key.outputShape), name: "output_5d")
        var biasTensor: MPSGraphTensor?
        if key.hasBias {
            let bias = graph.placeholder(shape: nums([1, 1, 1, 1, key.outputShape.c]), dataType: .float16, name: "bias")
            output = graph.addition(output, bias, name: "bias_add")
            biasTensor = bias
        }
        return Conv3DPlan(graph: graph, input: input, weight: weight, bias: biasTensor, output: output)
    }

    private func upsample2DPlan(_ key: Upsample2DKey) throws -> Upsample2DPlan {
        if let existing = upsample2DPlans[key] {
            return existing
        }
        let graph = MPSGraph()
        let input = graph.placeholder(shape: shape4BT(key.inputShape), dataType: .float16, name: "input")
        let weight = graph.placeholder(
            shape: nums([3, 3, key.inputShape.c, key.outputShape.c]),
            dataType: .float16,
            name: "weight"
        )

        let resizeSize = [Int32(key.outputShape.h), Int32(key.outputShape.w)]
        let resizeData = resizeSize.withUnsafeBufferPointer { Data(buffer: $0) }
        let size = graph.constant(resizeData, shape: nums([2]), dataType: .int32)
        let resized = graph.resizeNearest(
            input,
            sizeTensor: size,
            nearestRoundingMode: .floor,
            centerResult: false,
            alignCorners: false,
            layout: .NHWC,
            name: "nearest2x"
        )

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
            throw KVCraftMetalError.allocationFailed("MPSGraph 2D convolution descriptor")
        }
        var output = graph.convolution2D(resized, weights: weight, descriptor: desc, name: "conv2d")
        var biasTensor: MPSGraphTensor?
        if key.hasBias {
            let bias = graph.placeholder(shape: nums([1, 1, 1, key.outputShape.c]), dataType: .float16, name: "bias")
            output = graph.addition(output, bias, name: "bias_add")
            biasTensor = bias
        }

        let plan = Upsample2DPlan(graph: graph, input: input, weight: weight, bias: biasTensor, output: output)
        upsample2DPlans[key] = plan
        return plan
    }

    private func phaseUpsample2DPlan(_ key: PhaseUpsample2DKey) throws -> PhaseUpsample2DPlan {
        if let existing = phaseUpsample2DPlans[key] {
            return existing
        }
        let graph = MPSGraph()
        let input = graph.placeholder(shape: shape4BT(key.inputShape), dataType: .float16, name: "input")
        let weight = graph.placeholder(
            shape: nums([4, 2, 2, key.inputShape.c, key.outChannels]),
            dataType: .float16,
            name: "phase_weight"
        )
        let bias = key.hasBias ? graph.placeholder(shape: nums([1, 1, 1, key.outChannels]), dataType: .float16, name: "bias") : nil

        var phaseOutputs: [MPSGraphTensor] = []
        phaseOutputs.reserveCapacity(4)
        for phase in 0..<4 {
            let py = phase / 2
            let px = phase & 1
            guard let desc = MPSGraphConvolution2DOpDescriptor(
                strideInX: 1,
                strideInY: 1,
                dilationRateInX: 1,
                dilationRateInY: 1,
                groups: 1,
                paddingLeft: px == 0 ? 1 : 0,
                paddingRight: px == 0 ? 0 : 1,
                paddingTop: py == 0 ? 1 : 0,
                paddingBottom: py == 0 ? 0 : 1,
                paddingStyle: .explicit,
                dataLayout: .NHWC,
                weightsLayout: .HWIO
            ) else {
                throw KVCraftMetalError.allocationFailed("MPSGraph phase convolution descriptor")
            }
            let weightSlice = graph.sliceTensor(weight, dimension: 0, start: phase, length: 1, name: "phase_weight_\(phase)")
            let weight2D = graph.reshape(
                weightSlice,
                shape: nums([2, 2, key.inputShape.c, key.outChannels]),
                name: "phase_weight_2d_\(phase)"
            )
            var output = graph.convolution2D(input, weights: weight2D, descriptor: desc, name: "phase_conv_\(phase)")
            if let bias {
                output = graph.addition(output, bias, name: "phase_bias_\(phase)")
            }
            phaseOutputs.append(output)
        }
        let output = graph.concatTensors(phaseOutputs, dimension: 3, name: "phase_concat")
        let plan = PhaseUpsample2DPlan(graph: graph, input: input, weight: weight, bias: bias, output: output)
        phaseUpsample2DPlans[key] = plan
        return plan
    }

    private func attentionPlan(_ key: AttentionKey) -> AttentionPlan {
        if let existing = attentionPlans[key] {
            return existing
        }
        let graph = MPSGraph()
        let shape = nums([key.bt, 1, key.n, key.c])
        let q = graph.placeholder(shape: shape, dataType: .float16, name: "q")
        let k = graph.placeholder(shape: shape, dataType: .float16, name: "k")
        let v = graph.placeholder(shape: shape, dataType: .float16, name: "v")
        let scale = Float(1.0 / sqrt(Double(key.c)))
        let output = graph.scaledDotProductAttention(query: q, key: k, value: v, scale: scale, name: "sdpa")
        let plan = AttentionPlan(graph: graph, q: q, k: k, v: v, output: output)
        attentionPlans[key] = plan
        return plan
    }

    private func mpsCommandBuffer(_ commandBuffer: MTLCommandBuffer) -> MPSCommandBuffer {
        if let existing = commandBuffer as? MPSCommandBuffer {
            return existing
        }
        return MPSCommandBuffer(commandBuffer: commandBuffer)
    }
}

private struct Conv3DKey: Hashable {
    var inputShape: TensorShape
    var outputShape: TensorShape
    var kt: Int
    var kh: Int
    var kw: Int
    var paddingFront: Int
    var paddingBack: Int
    var paddingH: Int
    var paddingW: Int
    var hasBias: Bool
}

private final class Conv3DPlan {
    let graph: MPSGraph
    let input: MPSGraphTensor
    let weight: MPSGraphTensor
    let bias: MPSGraphTensor?
    let output: MPSGraphTensor

    init(graph: MPSGraph, input: MPSGraphTensor, weight: MPSGraphTensor, bias: MPSGraphTensor?, output: MPSGraphTensor) {
        self.graph = graph
        self.input = input
        self.weight = weight
        self.bias = bias
        self.output = output
    }
}

private struct Upsample2DKey: Hashable {
    var inputShape: TensorShape
    var outputShape: TensorShape
    var hasBias: Bool
}

private struct PhaseUpsample2DKey: Hashable {
    var inputShape: TensorShape
    var outChannels: Int
    var hasBias: Bool
}

private final class Upsample2DPlan {
    let graph: MPSGraph
    let input: MPSGraphTensor
    let weight: MPSGraphTensor
    let bias: MPSGraphTensor?
    let output: MPSGraphTensor

    init(graph: MPSGraph, input: MPSGraphTensor, weight: MPSGraphTensor, bias: MPSGraphTensor?, output: MPSGraphTensor) {
        self.graph = graph
        self.input = input
        self.weight = weight
        self.bias = bias
        self.output = output
    }
}

private final class PhaseUpsample2DPlan {
    let graph: MPSGraph
    let input: MPSGraphTensor
    let weight: MPSGraphTensor
    let bias: MPSGraphTensor?
    let output: MPSGraphTensor

    init(graph: MPSGraph, input: MPSGraphTensor, weight: MPSGraphTensor, bias: MPSGraphTensor?, output: MPSGraphTensor) {
        self.graph = graph
        self.input = input
        self.weight = weight
        self.bias = bias
        self.output = output
    }
}

private struct AttentionKey: Hashable {
    var bt: Int
    var n: Int
    var c: Int
}

private final class AttentionPlan {
    let graph: MPSGraph
    let q: MPSGraphTensor
    let k: MPSGraphTensor
    let v: MPSGraphTensor
    let output: MPSGraphTensor

    init(graph: MPSGraph, q: MPSGraphTensor, k: MPSGraphTensor, v: MPSGraphTensor, output: MPSGraphTensor) {
        self.graph = graph
        self.q = q
        self.k = k
        self.v = v
        self.output = output
    }
}

private func shape5(_ shape: TensorShape) -> [NSNumber] {
    nums([shape.b, shape.t, shape.h, shape.w, shape.c])
}

private func shape4BT(_ shape: TensorShape) -> [NSNumber] {
    nums([shape.b * shape.t, shape.h, shape.w, shape.c])
}

private func nums(_ values: [Int]) -> [NSNumber] {
    values.map { NSNumber(value: $0) }
}
