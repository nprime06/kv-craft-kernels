import Foundation
import Metal

struct WeightEntry: Decodable {
    let name: String
    let shape: [Int]
    let dtype: String
    let file: String
    let offset: Int
    let byteCount: Int
}

struct WeightManifest: Decodable {
    let format: String
    let source: String?
    let tensors: [WeightEntry]
}

public final class WeightArchive {
    private let context: MetalContext
    private var buffers: [String: MTLBuffer] = [:]
    private var foldedNearest2xWeights: [String: MTLBuffer] = [:]
    private var entries: [String: WeightEntry] = [:]
    private let zeroBuffer: MTLBuffer

    public init(url: URL, context: MetalContext) throws {
        self.context = context
        guard let zero = context.device.makeBuffer(length: 2, options: .storageModeShared) else {
            throw KVCraftMetalError.allocationFailed("zero bias")
        }
        zero.contents().storeBytes(of: UInt16(0), as: UInt16.self)
        self.zeroBuffer = zero

        let manifestURL = url.appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(WeightManifest.self, from: manifestData)
        guard manifest.format == "kvcraft-vae-decoder-f16-v1" else {
            throw KVCraftMetalError.malformedArchive("unsupported format \(manifest.format)")
        }
        for entry in manifest.tensors {
            guard entry.dtype == "float16" else {
                throw KVCraftMetalError.malformedArchive("\(entry.name) is \(entry.dtype), expected float16")
            }
            let fileURL = url.appendingPathComponent(entry.file)
            let data = try Data(contentsOf: fileURL)
            guard entry.offset >= 0, entry.offset + entry.byteCount <= data.count else {
                throw KVCraftMetalError.malformedArchive("\(entry.name) has an invalid byte range")
            }
            let slice = data.subdata(in: entry.offset..<(entry.offset + entry.byteCount))
            guard let buffer = context.device.makeBuffer(bytes: [UInt8](slice), length: slice.count, options: .storageModeShared) else {
                throw KVCraftMetalError.allocationFailed("weight \(entry.name)")
            }
            buffers[entry.name] = buffer
            entries[entry.name] = entry
        }
    }

    func buffer(_ name: String) throws -> MTLBuffer {
        guard let buffer = buffers[name] else {
            throw KVCraftMetalError.missingWeight(name)
        }
        return buffer
    }

    func optionalBuffer(_ name: String) -> MTLBuffer? {
        buffers[name]
    }

    func foldedNearest2xPhaseWeight(_ name: String) throws -> MTLBuffer {
        if let existing = foldedNearest2xWeights[name] {
            return existing
        }
        guard let source = buffers[name], let entry = entries[name] else {
            throw KVCraftMetalError.missingWeight(name)
        }
        let shape: [Int]
        if entry.shape.count == 5, entry.shape[0] == 1 {
            shape = Array(entry.shape.dropFirst())
        } else {
            shape = entry.shape
        }
        guard shape.count == 4, shape[0] == 3, shape[1] == 3 else {
            throw KVCraftMetalError.malformedArchive("\(name) must have shape [3, 3, Cin, Cout] or [1, 3, 3, Cin, Cout]")
        }
        let cin = shape[2]
        let cout = shape[3]
        let sourcePtr = source.contents().assumingMemoryBound(to: UInt16.self)
        var folded = [UInt16](repeating: 0, count: 4 * 2 * 2 * cin * cout)

        func sourceIndex(_ ky: Int, _ kx: Int, _ ci: Int, _ co: Int) -> Int {
            (((ky * 3 + kx) * cin + ci) * cout + co)
        }
        func foldedIndex(_ phase: Int, _ ky: Int, _ kx: Int, _ ci: Int, _ co: Int) -> Int {
            (((((phase * 2 + ky) * 2 + kx) * cin + ci) * cout + co))
        }
        func lowKernelIndex(phase: Int, highKernelIndex: Int) -> Int {
            if phase == 0 {
                return highKernelIndex == 0 ? 0 : 1
            }
            return highKernelIndex == 2 ? 1 : 0
        }

        for py in 0..<2 {
            for px in 0..<2 {
                let phase = py * 2 + px
                for ci in 0..<cin {
                    for co in 0..<cout {
                        var accum = [Float](repeating: 0, count: 4)
                        for ky in 0..<3 {
                            let ly = lowKernelIndex(phase: py, highKernelIndex: ky)
                            for kx in 0..<3 {
                                let lx = lowKernelIndex(phase: px, highKernelIndex: kx)
                                let dst = ly * 2 + lx
                                accum[dst] += Float(Float16(bitPattern: sourcePtr[sourceIndex(ky, kx, ci, co)]))
                            }
                        }
                        for ly in 0..<2 {
                            for lx in 0..<2 {
                                folded[foldedIndex(phase, ly, lx, ci, co)] = Float16(accum[ly * 2 + lx]).bitPattern
                            }
                        }
                    }
                }
            }
        }

        guard let buffer = folded.withUnsafeBytes({ raw in
            context.device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }) else {
            throw KVCraftMetalError.allocationFailed("folded upsample weight \(name)")
        }
        foldedNearest2xWeights[name] = buffer
        return buffer
    }

    func bias(_ name: String) -> (buffer: MTLBuffer, hasBias: Bool) {
        if let buffer = buffers[name] {
            return (buffer, true)
        }
        return (zeroBuffer, false)
    }
}
