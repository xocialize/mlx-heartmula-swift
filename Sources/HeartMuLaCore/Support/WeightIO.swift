// WeightIO.swift — safetensors in, key contract enforced (the mlx-yue2-swift / mlx-ace-step-swift idiom): load each shard on
// the CPU stream, cast floating tensors per the dtype policy, compare the module's flattened parameter paths with the on-disk
// keys — 0 missing, 0 unused, or throw — then `update(verify:)`. A partial load is the silent-failure class.
//
// Bundle layout = the V1 converter's (Tools/convert/convert.py): MLX-layout shards whose keys are the module paths (codec convs
// (O, K, I), weight_norm folded, SQ-Codec decoder renamed), `model.safetensors.index.json`, `config.json`, `tokenizer.json`.
import Foundation
import MLX
import MLXNN

public enum HeartMuLaError: Error, CustomStringConvertible {
    case missingFile(String)
    case keyContract(component: String, missing: [String], unused: [String])
    case badConfig(String)
    case invalidInput(String)
    case generation(String)
    case cancelled(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file: \(p)"
        case .keyContract(let c, let m, let u):
            return "\(c): key contract violated — missing \(m.count) \(m.prefix(6)), unused \(u.count) \(u.prefix(6))"
        case .badConfig(let s): return "bad config: \(s)"
        case .invalidInput(let s): return "invalid input: \(s)"
        case .generation(let s): return "generation failed: \(s)"
        case .cancelled(let s): return "cancelled: \(s)"
        }
    }
}

/// How floating tensors are cast at load. `.shipped` keeps the bundle dtypes (bf16 LM; fp32 or fp16 codec — the tier V0 ran);
/// `.float32` casts everything (the exact lane: the bf16 LM values in fp32 arithmetic, as the fp32 goldens were dumped).
public enum WeightDType: String, Sendable {
    case shipped
    case float32
}

public enum WeightIO {
    /// Loads every array of one safetensors file on the CPU stream (a lazy GPU-stream load of a multi-GB shard trips the Metal
    /// watchdog), casting floating tensors to `dtype` unless `keepFP32(key)`.
    public static func load(_ url: URL, dtype: DType?, keepFP32: (String) -> Bool = { _ in false }) throws -> [String: MLXArray] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw HeartMuLaError.missingFile(url.path) }
        return try Device.withDefaultDevice(.cpu) { () throws -> [String: MLXArray] in
            var out = [String: MLXArray]()
            for (k, v) in try loadArrays(url: url) {
                let floating = [DType.float16, .bfloat16, .float32].contains(v.dtype)
                if floating, let d = dtype, !keepFP32(k), v.dtype != d { out[k] = v.asType(d) } else { out[k] = v }
            }
            eval(Array(out.values))
            return out
        }
    }

    /// Every shard named by `model.safetensors.index.json` (or `model.safetensors` when there is no index).
    public static func shards(in dir: URL) throws -> [URL] {
        let index = dir.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: index.path) {
            let o = try ConfigIO.json(index)
            guard let map = o["weight_map"] as? [String: String] else { throw HeartMuLaError.badConfig("index has no weight_map") }
            return Set(map.values).sorted().map { dir.appendingPathComponent($0) }
        }
        let single = dir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: single.path) else { throw HeartMuLaError.missingFile(single.path) }
        return [single]
    }

    /// `rename` maps an on-disk key to the module's parameter path (identity for every HeartMuLa bundle).
    public static func apply(_ arrays: [String: MLXArray], to module: Module, component: String,
                             rename: (String) -> String = { $0 }) throws {
        var arrays = Dictionary(uniqueKeysWithValues: arrays.map { (rename($0.key), $0.value) })
        let expected = Set(module.parameters().flattened().map(\.0))
        let onDisk = Set(arrays.keys)
        let missing = expected.subtracting(onDisk).sorted()
        let unused = onDisk.subtracting(expected).sorted()
        guard missing.isEmpty && unused.isEmpty else {
            throw HeartMuLaError.keyContract(component: component, missing: missing, unused: unused)
        }
        try module.update(parameters: ModuleParameters.unflattened(arrays), verify: [.noUnusedKeys, .shapeMismatch])
        module.train(false)
        eval(module.parameters())
        arrays.removeAll()
    }
}

public enum ConfigIO {
    public static func load<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else { throw HeartMuLaError.missingFile(url.path) }
        return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
    }
    public static func json(_ url: URL) throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: url.path) else { throw HeartMuLaError.missingFile(url.path) }
        guard let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw HeartMuLaError.badConfig("\(url.lastPathComponent) is not a JSON object")
        }
        return o
    }
}
