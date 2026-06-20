import Darwin
import Foundation
import KVCraftVaeMetalDecoder

struct CLI {
    var weights: URL?
    var latentFile: URL?
    var udpPort: UInt16?
    var benchmarkIterations: Int?
    var warmupIterations = 2
    var latentHeight = 45
    var latentWidth = 80

    init(arguments: [String]) throws {
        var i = 1
        while i < arguments.count {
            let arg = arguments[i]
            func value() throws -> String {
                guard i + 1 < arguments.count else {
                    throw KVCraftMetalError.invalidArgument("missing value for \(arg)")
                }
                i += 1
                return arguments[i]
            }
            switch arg {
            case "--weights":
                weights = URL(fileURLWithPath: try value())
            case "--latent":
                latentFile = URL(fileURLWithPath: try value())
            case "--udp-port":
                guard let port = UInt16(try value()) else {
                    throw KVCraftMetalError.invalidArgument("invalid UDP port")
                }
                udpPort = port
            case "--benchmark":
                benchmarkIterations = Int(try value())
            case "--warmup":
                warmupIterations = Int(try value()) ?? warmupIterations
            case "--latent-height":
                latentHeight = Int(try value()) ?? latentHeight
            case "--latent-width":
                latentWidth = Int(try value()) ?? latentWidth
            case "--help", "-h":
                printUsageAndExit(0)
            default:
                throw KVCraftMetalError.invalidArgument("unknown argument \(arg)")
            }
            i += 1
        }
    }
}

func printUsageAndExit(_ code: Int32) -> Never {
    FileHandle.standardError.write(
        Data("""
        Usage:
          kvcraft-vae-metal --weights <archive-dir> --latent <raw-f16-latent>
          kvcraft-vae-metal --weights <archive-dir> --udp-port 7777
          kvcraft-vae-metal --weights <archive-dir> --benchmark 20 [--warmup 2] [--latent-height H --latent-width W]

        Latent payload format: one NHWTC frame, B=1 T=1 H=<latent-height> W=<latent-width> C=16, little-endian IEEE float16.
        Default latent size is 45x80, which decodes to 360x640 RGB. Output size is latent H/W multiplied by 8.
        The first decoded latent emits 1 RGB frame; later latents emit 4 RGB frames.

        """.utf8)
    )
    exit(code)
}

func runOne(decoder: KVCraftVaeDecoder, latentFile: URL) throws {
    let data = try Data(contentsOf: latentFile)
    let latent = try decoder.makeLatentTensor(bytes: data)
    let start = DispatchTime.now().uptimeNanoseconds
    let out = try decoder.decode(latent: latent)
    let end = DispatchTime.now().uptimeNanoseconds
    let ms = Double(end - start) / 1_000_000.0
    print("decoded \(out.shape.t)x \(out.shape.h)x\(out.shape.w)x\(out.shape.c) frame tensor in \(String(format: "%.2f", ms)) ms")
}

func runBenchmark(decoder: KVCraftVaeDecoder, latentFile: URL?, iterations: Int, warmup: Int) throws {
    guard iterations > 0, warmup >= 0 else {
        throw KVCraftMetalError.invalidArgument("benchmark iterations must be positive and warmup must be non-negative")
    }
    let expected = 1 * 1 * decoder.latentHeight * decoder.latentWidth * 16 * 2
    let payload: Data
    if let latentFile {
        payload = try Data(contentsOf: latentFile)
    } else {
        payload = Data(repeating: 0, count: expected)
    }
    let latent = try decoder.makeLatentTensor(bytes: payload)

    if warmup > 0 {
        decoder.reset()
        for _ in 0..<warmup {
            _ = try decoder.decode(latent: latent)
        }
    }

    decoder.reset()
    var outputFrames = 0
    var decodeMilliseconds: [Double] = []
    decodeMilliseconds.reserveCapacity(iterations)
    let startAll = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<iterations {
        let start = DispatchTime.now().uptimeNanoseconds
        let out = try decoder.decode(latent: latent)
        let end = DispatchTime.now().uptimeNanoseconds
        outputFrames += out.shape.t
        decodeMilliseconds.append(Double(end - start) / 1_000_000.0)
    }
    let endAll = DispatchTime.now().uptimeNanoseconds
    let seconds = Double(endAll - startAll) / 1_000_000_000.0
    let sorted = decodeMilliseconds.sorted()
    let mean = decodeMilliseconds.reduce(0, +) / Double(decodeMilliseconds.count)
    let p50 = sorted[sorted.count / 2]
    let p90 = sorted[min(sorted.count - 1, Int(Double(sorted.count - 1) * 0.90))]
    let fps = Double(outputFrames) / seconds
    print("iterations: \(iterations), warmup: \(warmup)")
    print("decoded frames: \(outputFrames), wall time: \(String(format: "%.3f", seconds)) s")
    print("fps: \(String(format: "%.2f", fps))")
    print("decode ms: mean \(String(format: "%.2f", mean)), p50 \(String(format: "%.2f", p50)), p90 \(String(format: "%.2f", p90))")
}

func runUDP(decoder: KVCraftVaeDecoder, port: UInt16) throws {
    let fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
    guard fd >= 0 else {
        throw KVCraftMetalError.invalidArgument("socket() failed")
    }
    defer { close(fd) }

    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.stride)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    addr.sin_addr = in_addr(s_addr: INADDR_ANY.bigEndian)
    let bindResult = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.stride))
        }
    }
    guard bindResult == 0 else {
        throw KVCraftMetalError.invalidArgument("bind() failed for UDP port \(port)")
    }

    let expected = 1 * 1 * decoder.latentHeight * decoder.latentWidth * 16 * 2
    var storage = [UInt8](repeating: 0, count: expected)
    print("listening on UDP \(port), expected datagram bytes: \(expected)")
    while true {
        let received = storage.withUnsafeMutableBytes { rawBuffer in
            recv(fd, rawBuffer.baseAddress, rawBuffer.count, 0)
        }
        if received != expected {
            fputs("dropped datagram with \(received) bytes; expected \(expected)\n", stderr)
            continue
        }
        let latent = try decoder.makeLatentTensor(bytes: Data(storage))
        let start = DispatchTime.now().uptimeNanoseconds
        let out = try decoder.decode(latent: latent)
        let end = DispatchTime.now().uptimeNanoseconds
        let ms = Double(end - start) / 1_000_000.0
        print("decoded \(out.shape.t) frame(s) in \(String(format: "%.2f", ms)) ms")
    }
}

do {
    let cli = try CLI(arguments: CommandLine.arguments)
    guard let weights = cli.weights else {
        printUsageAndExit(2)
    }
    let decoder = try KVCraftVaeDecoder(weightsURL: weights, latentHeight: cli.latentHeight, latentWidth: cli.latentWidth)
    if let iterations = cli.benchmarkIterations {
        try runBenchmark(decoder: decoder, latentFile: cli.latentFile, iterations: iterations, warmup: cli.warmupIterations)
    } else if let latentFile = cli.latentFile {
        try runOne(decoder: decoder, latentFile: latentFile)
    } else if let port = cli.udpPort {
        try runUDP(decoder: decoder, port: port)
    } else {
        printUsageAndExit(2)
    }
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
