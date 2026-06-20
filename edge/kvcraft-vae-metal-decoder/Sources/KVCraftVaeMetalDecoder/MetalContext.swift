import Foundation
import Metal

public final class MetalContext {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    public init(preferredDevice: MTLDevice? = nil) throws {
        guard let device = preferredDevice ?? MTLCreateSystemDefaultDevice() else {
            throw KVCraftMetalError.noMetalDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw KVCraftMetalError.allocationFailed("command queue")
        }
        #if SWIFT_PACKAGE
        let sourceURL = Bundle.module.url(
            forResource: "KVCraftVAE",
            withExtension: "metal",
            subdirectory: "Kernels"
        ) ?? Bundle.module.url(
            forResource: "KVCraftVAE",
            withExtension: "metal"
        )
        #else
        let sourceURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Kernels/KVCraftVAE.metal")
        #endif
        guard let sourceURL else {
            throw KVCraftMetalError.libraryLoadFailed("KVCraftVAE.metal was not found in package resources")
        }
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let options = MTLCompileOptions()
        options.mathMode = .fast
        do {
            self.library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw KVCraftMetalError.libraryLoadFailed(error.localizedDescription)
        }
        self.device = device
        self.queue = queue
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let existing = pipelines[name] {
            return existing
        }
        guard let function = library.makeFunction(name: name) else {
            throw KVCraftMetalError.pipelineFailed(name)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }
}
