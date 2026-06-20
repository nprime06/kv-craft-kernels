import Foundation
import Metal
import MetalPerformanceShaders

struct ProbeError: Error, CustomStringConvertible {
    let description: String
}

struct CLI {
    var iterations = 30
    var warmup = 5
    var cases: [GemmCase] = [
        GemmCase(name: "hires_res_conv_tile_4096", m: 4096, k: 3 * 3 * 3 * 96, n: 96),
        GemmCase(name: "hires_res_conv_tile_8192", m: 8192, k: 3 * 3 * 3 * 96, n: 96),
        GemmCase(name: "mid_res_conv_tile_4096", m: 4096, k: 3 * 3 * 3 * 192, n: 192),
        GemmCase(name: "bottleneck_conv_tile_4096", m: 4096, k: 3 * 3 * 3 * 384, n: 384),
    ]

    init(arguments: [String]) throws {
        var i = 1
        while i < arguments.count {
            let arg = arguments[i]
            func value() throws -> String {
                guard i + 1 < arguments.count else {
                    throw ProbeError(description: "missing value for \(arg)")
                }
                i += 1
                return arguments[i]
            }

            switch arg {
            case "--iterations":
                iterations = Int(try value()) ?? iterations
            case "--warmup":
                warmup = Int(try value()) ?? warmup
            case "--case":
                let fields = try value().split(separator: ",")
                guard fields.count == 4,
                      let m = Int(fields[1]),
                      let k = Int(fields[2]),
                      let n = Int(fields[3]) else {
                    throw ProbeError(description: "--case expects name,m,k,n")
                }
                cases = [GemmCase(name: String(fields[0]), m: m, k: k, n: n)]
            case "--append-case":
                let fields = try value().split(separator: ",")
                guard fields.count == 4,
                      let m = Int(fields[1]),
                      let k = Int(fields[2]),
                      let n = Int(fields[3]) else {
                    throw ProbeError(description: "--append-case expects name,m,k,n")
                }
                cases.append(GemmCase(name: String(fields[0]), m: m, k: k, n: n))
            case "--help", "-h":
                printUsageAndExit(0)
            default:
                throw ProbeError(description: "unknown argument \(arg)")
            }
            i += 1
        }
        guard iterations > 0, warmup >= 0 else {
            throw ProbeError(description: "iterations must be positive and warmup must be non-negative")
        }
    }
}

struct GemmCase {
    let name: String
    let m: Int
    let k: Int
    let n: Int

    var macs: Double {
        Double(m) * Double(k) * Double(n)
    }
}

struct TimingStats {
    let meanMs: Double
    let p50Ms: Double
    let p90Ms: Double
}

struct VariantTiming {
    let name: String
    let stats: TimingStats
}

func printUsageAndExit(_ code: Int32) -> Never {
    FileHandle.standardError.write(
        Data("""
        Usage:
          quant-gemm-probe [--iterations 30] [--warmup 5]
          quant-gemm-probe --case name,m,k,n

        Cases are matrix multiplications C[M,N] = A[M,K] * B[K,N].
        The default cases approximate im2col tiles for KVCraft VAE decoder 3x3x3 convs.

        """.utf8)
    )
    exit(code)
}

func makeDescriptor(_ dataType: MPSDataType, _ rows: Int, _ cols: Int) -> MPSNDArrayDescriptor {
    let descriptor = MPSNDArrayDescriptor(dataType: dataType, shape: [NSNumber(value: rows), NSNumber(value: cols)])
    if #available(macOS 15.0, *) {
        descriptor.preferPackedRows = true
    }
    return descriptor
}

func makeBuffer(device: MTLDevice, byteCount: Int, pattern: UInt8) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: byteCount, options: [.storageModeShared]) else {
        throw ProbeError(description: "failed to allocate \(byteCount) byte buffer")
    }
    memset(buffer.contents(), Int32(pattern), byteCount)
    return buffer
}

func makeNDArray(device: MTLDevice, dataType: MPSDataType, rows: Int, cols: Int, byteCount: Int, pattern: UInt8) throws -> MPSNDArray {
    let buffer = try makeBuffer(device: device, byteCount: byteCount, pattern: pattern)
    return MPSNDArray(buffer: buffer, offset: 0, descriptor: makeDescriptor(dataType, rows, cols))
}

func quantizedKernel(
    device: MTLDevice,
    left: MPSNDArrayQuantizationDescriptor?,
    right: MPSNDArrayQuantizationDescriptor?
) -> MPSNDArrayQuantizedMatrixMultiplication {
    let kernel = MPSNDArrayQuantizedMatrixMultiplication(
        device: device,
        leftQuantizationDescriptor: left,
        rightQuantizationDescriptor: right
    )
    kernel.beta = 0.0
    return kernel
}

func runTimed(iterations: Int, warmup: Int, encode: () throws -> MTLCommandBuffer) throws -> TimingStats {
    for _ in 0..<warmup {
        let commandBuffer = try encode()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
    }

    var ms: [Double] = []
    ms.reserveCapacity(iterations)
    for _ in 0..<iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        let commandBuffer = try encode()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        let end = DispatchTime.now().uptimeNanoseconds
        ms.append(Double(end - start) / 1_000_000.0)
    }
    let sorted = ms.sorted()
    let mean = ms.reduce(0, +) / Double(ms.count)
    return TimingStats(
        meanMs: mean,
        p50Ms: sorted[sorted.count / 2],
        p90Ms: sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.90))]
    )
}

@available(macOS 15.0, *)
func probe(case gemm: GemmCase, device: MTLDevice, queue: MTLCommandQueue, iterations: Int, warmup: Int) throws {
    let fp16A = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.k, byteCount: gemm.m * gemm.k * 2, pattern: 0x3c)
    let fp16B = try makeNDArray(device: device, dataType: .float16, rows: gemm.k, cols: gemm.n, byteCount: gemm.k * gemm.n * 2, pattern: 0x2a)
    let fp16C = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.n, byteCount: gemm.m * gemm.n * 2, pattern: 0)

    let int8A = try makeNDArray(device: device, dataType: .int8, rows: gemm.m, cols: gemm.k, byteCount: gemm.m * gemm.k, pattern: 1)
    let int8B = try makeNDArray(device: device, dataType: .int8, rows: gemm.k, cols: gemm.n, byteCount: gemm.k * gemm.n, pattern: 1)
    let int8C = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.n, byteCount: gemm.m * gemm.n * 2, pattern: 0)
    let uint8A = try makeNDArray(device: device, dataType: .uInt8, rows: gemm.m, cols: gemm.k, byteCount: gemm.m * gemm.k, pattern: 1)
    let uint8B = try makeNDArray(device: device, dataType: .uInt8, rows: gemm.k, cols: gemm.n, byteCount: gemm.k * gemm.n, pattern: 1)
    let uint8C = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.n, byteCount: gemm.m * gemm.n * 2, pattern: 0)
    let w8C = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.n, byteCount: gemm.m * gemm.n * 2, pattern: 0)
    let a8C = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.n, byteCount: gemm.m * gemm.n * 2, pattern: 0)
    let int4A = try makeNDArray(device: device, dataType: .int4, rows: gemm.m, cols: gemm.k, byteCount: (gemm.m * gemm.k + 1) / 2, pattern: 0x11)
    let int4B = try makeNDArray(device: device, dataType: .int4, rows: gemm.k, cols: gemm.n, byteCount: (gemm.k * gemm.n + 1) / 2, pattern: 0x11)
    let int4C = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.n, byteCount: gemm.m * gemm.n * 2, pattern: 0)
    let w4C = try makeNDArray(device: device, dataType: .float16, rows: gemm.m, cols: gemm.n, byteCount: gemm.m * gemm.n * 2, pattern: 0)

    let scaleA = MPSNDArray(device: device, scalar: 0.01)
    let scaleB = MPSNDArray(device: device, scalar: 0.01)

    let fp16Kernel = MPSNDArrayMatrixMultiplication(device: device, sourceCount: 2)
    fp16Kernel.beta = 0.0

    let int8DescA = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: false, hasMinValue: false)
    let int8DescB = MPSNDArrayAffineQuantizationDescriptor(dataType: .int8, hasZeroPoint: false, hasMinValue: false)
    let uint8DescA = MPSNDArrayAffineQuantizationDescriptor(dataType: .uInt8, hasZeroPoint: false, hasMinValue: false)
    let uint8DescB = MPSNDArrayAffineQuantizationDescriptor(dataType: .uInt8, hasZeroPoint: false, hasMinValue: false)
    let int4DescA = MPSNDArrayAffineQuantizationDescriptor(dataType: .int4, hasZeroPoint: false, hasMinValue: false)
    let int4DescB = MPSNDArrayAffineQuantizationDescriptor(dataType: .int4, hasZeroPoint: false, hasMinValue: false)
    let int8Kernel = quantizedKernel(device: device, left: int8DescA, right: int8DescB)
    let uint8Kernel = quantizedKernel(device: device, left: uint8DescA, right: uint8DescB)
    let weightOnlyKernel = quantizedKernel(device: device, left: nil, right: int8DescB)
    let activationOnlyKernel = quantizedKernel(device: device, left: int8DescA, right: nil)
    let int4Kernel = quantizedKernel(device: device, left: int4DescA, right: int4DescB)
    let weight4Kernel = quantizedKernel(device: device, left: nil, right: int4DescB)

    var variants: [VariantTiming] = []

    variants.append(VariantTiming(name: "fp16_mps_ndarray", stats: try runTimed(iterations: iterations, warmup: warmup) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError(description: "failed to make command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError(description: "failed to make compute encoder")
        }
        fp16Kernel.encode(to: encoder, commandBuffer: commandBuffer, sourceArrays: [fp16A, fp16B], destinationArray: fp16C)
        encoder.endEncoding()
        return commandBuffer
    }))

    variants.append(VariantTiming(name: "int8_affine_both", stats: try runTimed(iterations: iterations, warmup: warmup) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError(description: "failed to make command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError(description: "failed to make compute encoder")
        }
        int8Kernel.encode(to: encoder, commandBuffer: commandBuffer, sourceArrays: [int8A, int8B, scaleA, scaleB], destinationArray: int8C)
        encoder.endEncoding()
        return commandBuffer
    }))

    variants.append(VariantTiming(name: "uint8_affine_both", stats: try runTimed(iterations: iterations, warmup: warmup) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError(description: "failed to make command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError(description: "failed to make compute encoder")
        }
        uint8Kernel.encode(to: encoder, commandBuffer: commandBuffer, sourceArrays: [uint8A, uint8B, scaleA, scaleB], destinationArray: uint8C)
        encoder.endEncoding()
        return commandBuffer
    }))

    variants.append(VariantTiming(name: "fp16_x_int8_weight_only", stats: try runTimed(iterations: iterations, warmup: warmup) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError(description: "failed to make command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError(description: "failed to make compute encoder")
        }
        weightOnlyKernel.encode(to: encoder, commandBuffer: commandBuffer, sourceArrays: [fp16A, int8B, scaleB], destinationArray: w8C)
        encoder.endEncoding()
        return commandBuffer
    }))

    variants.append(VariantTiming(name: "int8_activation_x_fp16", stats: try runTimed(iterations: iterations, warmup: warmup) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError(description: "failed to make command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError(description: "failed to make compute encoder")
        }
        activationOnlyKernel.encode(to: encoder, commandBuffer: commandBuffer, sourceArrays: [int8A, fp16B, scaleA], destinationArray: a8C)
        encoder.endEncoding()
        return commandBuffer
    }))

    variants.append(VariantTiming(name: "int4_affine_both", stats: try runTimed(iterations: iterations, warmup: warmup) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError(description: "failed to make command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError(description: "failed to make compute encoder")
        }
        int4Kernel.encode(to: encoder, commandBuffer: commandBuffer, sourceArrays: [int4A, int4B, scaleA, scaleB], destinationArray: int4C)
        encoder.endEncoding()
        return commandBuffer
    }))

    variants.append(VariantTiming(name: "fp16_x_int4_weight_only", stats: try runTimed(iterations: iterations, warmup: warmup) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError(description: "failed to make command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError(description: "failed to make compute encoder")
        }
        weight4Kernel.encode(to: encoder, commandBuffer: commandBuffer, sourceArrays: [fp16A, int4B, scaleB], destinationArray: w4C)
        encoder.endEncoding()
        return commandBuffer
    }))

    func tops(_ stats: TimingStats) -> Double {
        let seconds = stats.p50Ms / 1_000.0
        return gemm.macs / seconds / 1_000_000_000_000.0
    }

    let fp16P50 = variants[0].stats.p50Ms
    print("case: \(gemm.name) M=\(gemm.m) K=\(gemm.k) N=\(gemm.n)")
    for variant in variants {
        print("  \(variant.name): mean \(String(format: "%.3f", variant.stats.meanMs)) ms, p50 \(String(format: "%.3f", variant.stats.p50Ms)) ms, p90 \(String(format: "%.3f", variant.stats.p90Ms)) ms, effective \(String(format: "%.2f", tops(variant.stats))) TMAC/s, vs fp16 \(String(format: "%.2fx", fp16P50 / variant.stats.p50Ms))")
    }
}

do {
    let cli = try CLI(arguments: CommandLine.arguments)
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw ProbeError(description: "no Metal device")
    }
    guard let queue = device.makeCommandQueue() else {
        throw ProbeError(description: "failed to make command queue")
    }

    if #available(macOS 15.0, *) {
        print("device: \(device.name)")
        print("iterations: \(cli.iterations), warmup: \(cli.warmup)")
        for gemm in cli.cases {
            try autoreleasepool {
                try probe(case: gemm, device: device, queue: queue, iterations: cli.iterations, warmup: cli.warmup)
            }
        }
    } else {
        throw ProbeError(description: "MPS quantized matrix multiplication requires macOS 15+")
    }
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
