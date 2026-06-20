// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "KVCraftVaeMetalDecoder",
    platforms: [.macOS("15.0")],
    products: [
        .library(name: "KVCraftVaeMetalDecoder", targets: ["KVCraftVaeMetalDecoder"]),
        .executable(name: "kvcraft-vae-metal", targets: ["KVCraftVaeMetalCLI"]),
        .executable(name: "quant-gemm-probe", targets: ["QuantGemmProbe"]),
    ],
    targets: [
        .target(
            name: "KVCraftVaeMetalDecoder",
            resources: [.process("Kernels")],
            linkerSettings: [
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
            ]
        ),
        .executableTarget(
            name: "KVCraftVaeMetalCLI",
            dependencies: ["KVCraftVaeMetalDecoder"]
        ),
        .executableTarget(
            name: "QuantGemmProbe",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
            ]
        ),
    ]
)
