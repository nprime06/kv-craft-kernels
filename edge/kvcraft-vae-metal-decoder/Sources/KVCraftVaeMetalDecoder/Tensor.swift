import Foundation
import Metal

public struct TensorShape: Hashable, Sendable {
    public var b: Int
    public var t: Int
    public var h: Int
    public var w: Int
    public var c: Int

    public init(_ b: Int, _ t: Int, _ h: Int, _ w: Int, _ c: Int) {
        self.b = b
        self.t = t
        self.h = h
        self.w = w
        self.c = c
    }

    public var elementCount: Int { b * t * h * w * c }
    public var byteCountF16: Int { elementCount * MemoryLayout<UInt16>.stride }

    var tensorParams: TensorParams {
        TensorParams(
            b: UInt32(b), t: UInt32(t), h: UInt32(h), w: UInt32(w), c: UInt32(c)
        )
    }
}

public final class GpuTensor {
    public let shape: TensorShape
    public let buffer: MTLBuffer

    public init(device: MTLDevice, shape: TensorShape, options: MTLResourceOptions = .storageModePrivate) throws {
        self.shape = shape
        guard let buffer = device.makeBuffer(length: max(1, shape.byteCountF16), options: options) else {
            throw KVCraftMetalError.allocationFailed("buffer \(shape)")
        }
        self.buffer = buffer
    }

    public init(buffer: MTLBuffer, shape: TensorShape) {
        self.shape = shape
        self.buffer = buffer
    }
}

public enum KVCraftMetalError: Error, CustomStringConvertible {
    case noMetalDevice
    case libraryLoadFailed(String)
    case pipelineFailed(String)
    case allocationFailed(String)
    case missingWeight(String)
    case malformedArchive(String)
    case invalidArgument(String)

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device is available"
        case .libraryLoadFailed(let message):
            return "Metal library load failed: \(message)"
        case .pipelineFailed(let name):
            return "Failed to create Metal pipeline \(name)"
        case .allocationFailed(let message):
            return "Allocation failed: \(message)"
        case .missingWeight(let name):
            return "Missing weight tensor \(name)"
        case .malformedArchive(let message):
            return "Malformed weight archive: \(message)"
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        }
    }
}

struct Conv3DParams {
    var b: UInt32
    var t: UInt32
    var h: UInt32
    var w: UInt32
    var cin: UInt32
    var cout: UInt32
    var kt: UInt32
    var kh: UInt32
    var kw: UInt32
    var padT: Int32
    var padH: Int32
    var padW: Int32
    var cacheT: UInt32
    var cacheValid: UInt32
    var hasBias: UInt32
}

struct NormParams {
    var b: UInt32
    var t: UInt32
    var h: UInt32
    var w: UInt32
    var c: UInt32
    var applySilu: UInt32
    var hasBias: UInt32
}

struct Up2DConvParams {
    var b: UInt32
    var t: UInt32
    var h: UInt32
    var w: UInt32
    var cin: UInt32
    var cout: UInt32
    var kh: UInt32
    var kw: UInt32
    var padH: Int32
    var padW: Int32
    var hasBias: UInt32
}

struct CausalSourceParams {
    var b: UInt32
    var t: UInt32
    var h: UInt32
    var w: UInt32
    var c: UInt32
    var preT: UInt32
    var cacheT: UInt32
    var cacheValid: UInt32
}

struct AttentionSplitParams {
    var bt: UInt32
    var n: UInt32
    var c: UInt32
}

struct TensorParams {
    var b: UInt32
    var t: UInt32
    var h: UInt32
    var w: UInt32
    var c: UInt32
}
