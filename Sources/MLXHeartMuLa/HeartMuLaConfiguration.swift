// HeartMuLaConfiguration.swift — init-time configuration for `HeartMuLaPackage` (C9). Per-request caption / lyrics / length /
// seed ride the canonical `MusicRequest`.
//
// TWO weight sets back the model, each one of the V1 / V2b converter's bundles (`Tools/convert/convert.py`, `heartmula-gates
// quantize`): the LM (`config.json`, the safetensors, `tokenizer.json`, `gen_config.json`) — bf16 as converted, or PRE-QUANTISED
// 8-bit (the 217 layer Linears affine group 64, the `quantization` block in its config; nothing is quantised at load) — and the
// codec (`config.json`, the safetensors) at fp32 or fp16. Every bundle carries heartlib's Apache-2.0 LICENSE and a NOTICE with
// the grant chain. The four repos below are the V4 publication (AB-D-0203): hash-equal to the gated bundles and pinned to their
// published commits, so a fresh machine's engine materializes exactly what V1–V3 gated; `lmDirectory` / `codecDirectory` stay
// the dev escape hatch.

import Foundation
import HeartMuLaCore
import MLXToolKit

/// The codec's precision — the second tier axis (the LM's is `quant`). Never bf16: upstream's README says bf16 degrades audio.
public enum HeartMuLaCodecPrecision: String, Sendable, Codable, CaseIterable {
    /// The default (V2b C16 PASS): lyric and aesthetic scores unchanged on the same codes, 23.6 dB from fp32 (rounding-grid
    /// flips), 3.5× faster per window, 3.1 GB smaller.
    case fp16
    /// The exact codec (V2 parity 115–118 dB vs torch); needs TF32 off for the whole process (AB-L-0315).
    case fp32
}

public struct HeartMuLaConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The LM's precision: `.int8` (the pre-quantised 8-bit bundle, the default) or `.bf16`. Nothing else is offered.
    public var quant: Quant
    public var codec: HeartMuLaCodecPrecision
    /// Revision of the LM repo / the codec repo; nil = the published, hash-verified commit (`lmRevisionPin` / `codecRevisionPin`).
    public var revision: String?
    public var codecRevision: String?
    /// Explicit LM bundle directory (the converter's layout) — the dev escape hatch, never touches the network.
    public var lmDirectory: URL?
    /// Explicit codec bundle directory.
    public var codecDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?
    /// MLX buffer-cache limit while the model is resident (the fleet's bounded-cache rule; every footprint was measured at 1 GiB).
    public var cacheLimitBytes: UInt64

    public init(quant: Quant = .int8, codec: HeartMuLaCodecPrecision = .fp16, revision: String? = nil, codecRevision: String? = nil,
                lmDirectory: URL? = nil, codecDirectory: URL? = nil, modelsRootDirectory: URL? = nil, cacheLimitBytes: UInt64 = 1 << 30) {
        self.quant = quant; self.codec = codec; self.revision = revision; self.codecRevision = codecRevision
        self.lmDirectory = lmDirectory; self.codecDirectory = codecDirectory
        self.modelsRootDirectory = modelsRootDirectory; self.cacheLimitBytes = cacheLimitBytes
    }

    // MARK: repos (published V4, AB-D-0203; V1 converted from HeartMuLa's own repos)

    public static let lmBF16Repo = "mlx-community/HeartMuLa-oss-3B-bf16"
    public static let lm8bitRepo = "mlx-community/HeartMuLa-oss-3B-8bit"
    public static let codecFP32Repo = "mlx-community/HeartCodec-oss-fp32"
    public static let codecFP16Repo = "mlx-community/HeartCodec-oss-fp16"
    /// The published commits (V4), each hash-equal to the gated bundle.
    public static let lmBF16RevisionPin = "625d8d0edb27b652cbdaba5c6530b8d2ffe29868"
    public static let lm8bitRevisionPin = "ff4fbf65ed698d58049674c6dc8e0141bfb64b60"
    public static let codecFP32RevisionPin = "a03fdf4c3735863c347e618b280ebd27a1478c90"
    public static let codecFP16RevisionPin = "5865bd08de864711a09b40a30fc2208fcb8def62"
    /// The upstream checkpoints the converter read (V1 gated against them).
    public static let upstreamModelRepo = "HeartMuLa/HeartMuLa-oss-3B-happy-new-year"
    public static let upstreamModelRevision = "41f6fc68490e11dc43fdabaa6b5767946408c903"
    public static let upstreamCodecRepo = "HeartMuLa/HeartCodec-oss-20260123"
    public static let upstreamCodecRevision = "f889dab0532cfa4bf459f2a3367eb6d346b8eeda"
    public static let upstreamGenRepo = "HeartMuLa/HeartMuLaGen"
    public static let upstreamGenRevision = "2e18e01702011f4f7dc8642b6260967df62417ea"

    /// The LM repo for `quant`; nil for a quant no bundle exists for (refused by `load()`).
    public var lmRepo: String? {
        switch quant {
        case .bf16: Self.lmBF16Repo
        case .int8: Self.lm8bitRepo
        default: nil
        }
    }

    public var codecRepo: String {
        switch codec {
        case .fp32: Self.codecFP32Repo
        case .fp16: Self.codecFP16Repo
        }
    }

    /// The pinned commit of `lmRepo` (nil for a quant with no bundle).
    public var lmRevisionPin: String? {
        switch quant {
        case .bf16: Self.lmBF16RevisionPin
        case .int8: Self.lm8bitRevisionPin
        default: nil
        }
    }

    public var codecRevisionPin: String {
        switch codec {
        case .fp32: Self.codecFP32RevisionPin
        case .fp16: Self.codecFP16RevisionPin
        }
    }

    public var effectiveRevision: String? { revision ?? lmRevisionPin }
    public var effectiveCodecRevision: String { codecRevision ?? codecRevisionPin }

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey { case quant, codec, revision, codecRevision, cacheLimitBytes }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        quant = try c.decodeIfPresent(Quant.self, forKey: .quant) ?? .int8
        codec = try c.decodeIfPresent(HeartMuLaCodecPrecision.self, forKey: .codec) ?? .fp16
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        codecRevision = try c.decodeIfPresent(String.self, forKey: .codecRevision)
        cacheLimitBytes = try c.decodeIfPresent(UInt64.self, forKey: .cacheLimitBytes) ?? (1 << 30)
    }
}

// MARK: - Footprint hint (the codec axis at the same LM quant)

extension HeartMuLaConfiguration: FootprintConfigured {
    /// The manifest's `QuantFootprint`s are the fp16-codec pairings; the fp32 codec is 3.1 GB larger resident and decodes with a
    /// larger activation, so those pairings hint their own measured bytes.
    public var residentBytesHint: UInt64? {
        guard codec == .fp32 else { return nil }
        switch quant {
        case .bf16: return HeartMuLaPackage.bf16FP32ResidentBytes
        case .int8: return HeartMuLaPackage.int8FP32ResidentBytes
        default: return nil
        }
    }

    public var peakActivationBytesHint: UInt64? {
        guard codec == .fp32 else { return nil }
        switch quant {
        case .bf16: return HeartMuLaPackage.bf16FP32PeakActivationBytes
        case .int8: return HeartMuLaPackage.int8FP32PeakActivationBytes
        default: return nil
        }
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension HeartMuLaConfiguration: WeightSourcing {
    /// The files `HeartMuLaPipeline` opens per bundle — the materialization contract. The bf16 LM and fp32 codec are sharded:
    /// `WeightIO.shards` reads `model.safetensors.index.json` to find them, so `*.json` must carry it (v0.1.0 declared only
    /// `config.json` / `tokenizer.json` / `gen_config.json` and an engine download of a sharded repo failed — V4); the 8-bit LM and
    /// fp16 codec are one file. `*.json` also covers the configs and the tokenizer.
    public static let lmFiles = ["*.json", "*.safetensors"]
    public static let codecFiles = ["*.json", "*.safetensors"]
    /// Representative file per bundle for the missing-probe.
    static let lmProbe = "tokenizer.json"
    static let codecProbe = "config.json"

    public var weightSources: [WeightSource] {
        [WeightSource(role: "lm", repo: lmRepo ?? Self.lm8bitRepo, revision: effectiveRevision ?? Self.lm8bitRevisionPin, matching: Self.lmFiles),
         WeightSource(role: "codec", repo: codecRepo, revision: effectiveCodecRevision, matching: Self.codecFiles)]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if lmDirectory != nil || codecDirectory != nil {
            let fm = FileManager.default
            let present = [lmDirectory.map { fm.fileExists(atPath: $0.appendingPathComponent(Self.lmProbe).path) } ?? false,
                           codecDirectory.map { fm.fileExists(atPath: $0.appendingPathComponent(Self.codecProbe).path) } ?? false]
            return zip(weightSources, present).filter { !$0.1 }.map(\.0)
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// The two bundle directories `load()` opens: the explicit ones, else each repo's snapshot (hub-client layout) or flat directory
    /// (engine-executed layout) under the store. Nil when either does not resolve.
    public func resolvedBundles(storeRoot: URL?) -> (lm: URL, codec: URL)? {
        let store = ModelStore(root: storeRoot)
        func dir(_ explicit: URL?, _ repo: String, _ revision: String?, _ probe: String) -> URL? {
            if let explicit { return explicit }
            if let snapshot = store.snapshotDirectory(for: repo, revision: revision),
               FileManager.default.fileExists(atPath: snapshot.appendingPathComponent(probe).path) { return snapshot }
            if let flat = store.directory(for: repo),
               FileManager.default.fileExists(atPath: flat.appendingPathComponent(probe).path) { return flat }
            return nil
        }
        guard let lmRepo,
              let l = dir(lmDirectory, lmRepo, effectiveRevision, Self.lmProbe),
              let c = dir(codecDirectory, codecRepo, effectiveCodecRevision, Self.codecProbe) else { return nil }
        return (l, c)
    }
}

// MARK: - Cold-start prewarm

extension HeartMuLaConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] { resolvedBundles(storeRoot: modelsRootDirectory).map { [$0.lm, $0.codec] } ?? [] }
}
