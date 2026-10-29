// HeartMuLaPipeline.swift — `HeartMuLaGenPipeline` end to end: preprocess → _forward (the AR rollout) → HeartCodec.detokenize →
// the output limiter. Loads the two bundles the converter writes (`HeartMuLa-oss-3B-bf16`, `HeartCodec-oss-{fp32,fp16}`)
// through the key contract (WeightIO). The limiter is the port's addition (V0 outputs peaked at 1.03–1.37 on 8 of 12 songs):
// one gain `min(1, 0.891 / peak)` over the whole song — a −1 dBFS ceiling, no compressor, no per-sample clipping.
import Foundation
import MLX
import MLXNN
import MLXRandom

public struct SongRequest: Sendable {
    public var tags: String
    public var lyrics: String
    public var maxSeconds: Double
    public var seed: UInt64
    public var sampling: SamplingConfig
    public var codecSteps: Int
    public var codecGuidance: Float
    /// false only for the V3 envelope lane (run to the cap whatever the model says).
    public var stopOnEndToken: Bool
    public init(tags: String, lyrics: String, maxSeconds: Double = 120, seed: UInt64 = 0, sampling: SamplingConfig = SamplingConfig(),
                codecSteps: Int = 10, codecGuidance: Float = 1.25, stopOnEndToken: Bool = true) {
        self.tags = tags; self.lyrics = lyrics; self.maxSeconds = maxSeconds; self.seed = seed; self.sampling = sampling
        self.codecSteps = codecSteps; self.codecGuidance = codecGuidance; self.stopOnEndToken = stopOnEndToken
    }

    /// heartlib's `max_audio_length_ms`, rounded to the millisecond.
    public var maxMs: Int { Int((maxSeconds * 1000).rounded()) }
    /// LM frames the cap allows: the first frame plus `maxMs / 80` more (heartlib's `max_audio_frames`) — so heartlib's 120 s
    /// cap yields up to 1 501 frames, 120.08 s. A host that wants `maxSeconds` to bound the audio passes
    /// `SongRequest.lmSeconds(cappingAudioAt:)`.
    public var maxFrames: Int { maxMs / 80 + 1 }

    /// The `maxSeconds` whose rollout yields at most `floor(seconds × 12.5)` frames, i.e. audio no longer than `seconds`.
    public static func lmSeconds(cappingAudioAt seconds: Double) -> Double {
        Double(max(1, Int((seconds * 12.5).rounded(.down))) - 1) * 0.08
    }
}

public struct SongResult {
    public let audio: MLXArray          // (2, samples) fp32, limited
    public let codes: MLXArray          // (8, F) int32 — the rollout the codec decoded
    public let sampleRate: Int
    public let frames: Int
    public let endedOnEndToken: Bool
    public let effectiveSeconds: Double
    public let rawPeak: Float
    public let limiterGain: Float
    public let promptTokens: Int
    public let lmSeconds: Double
    public let prefillSeconds: Double
    public let ditSeconds: [Double]
    public let sqSeconds: Double
}

public enum Limiter {
    public static let ceiling: Float = 0.891          // −1 dBFS
    /// → (limited audio, raw peak, gain)
    public static func apply(_ audio: MLXArray) -> (MLXArray, Float, Float) {
        let peak = abs(audio).max().item(Float.self)
        let gain = peak > ceiling ? ceiling / peak : 1
        return (gain < 1 ? audio * gain : audio, peak, gain)
    }
}

public final class HeartMuLaPipeline {
    public let lm: HeartMuLaModel
    public let codec: HeartCodecModel
    public let tokenizer: BPETokenizer
    public let genConfig: GenConfig
    /// The LM bundle's `quantization` block (pre-quantised 8- / 4-bit tiers), nil for the bf16 bundle.
    public let lmQuantization: LMQuantization?

    /// `lmDir` = the LM bundle (weights, config.json, tokenizer.json, gen_config.json); `codecDir` = a codec bundle.
    /// `.float32` casts every tensor to fp32 (the exact lane); `.shipped` keeps the bundle dtypes.
    public init(lmDir: URL, codecDir: URL, dtype: WeightDType = .shipped) throws {
        let lc = try ConfigIO.load(HeartMuLaConfig.self, from: lmDir.appendingPathComponent("config.json"))
        let cc = try ConfigIO.load(CodecConfig.self, from: codecDir.appendingPathComponent("config.json"))
        tokenizer = try BPETokenizer(file: lmDir.appendingPathComponent("tokenizer.json"))
        genConfig = (try? ConfigIO.load(GenConfig.self, from: lmDir.appendingPathComponent("gen_config.json"))) ?? GenConfig()
        lm = try HeartMuLaModel(lc)
        codec = HeartCodecModel(cc)
        lmQuantization = try LMQuantization.from(config: lmDir.appendingPathComponent("config.json"))
        if let q = lmQuantization { q.apply(to: lm) }
        let cast: DType? = dtype == .float32 && lmQuantization == nil ? .float32 : nil
        try Self.load(lm, from: lmDir, dtype: cast, component: "lm")
        try Self.load(codec, from: codecDir, dtype: cast, component: "codec")
    }

    static func load(_ module: Module, from dir: URL, dtype: DType?, component: String) throws {
        var all = [String: MLXArray]()
        for shard in try WeightIO.shards(in: dir) {
            for (k, v) in try WeightIO.load(shard, dtype: dtype) { all[k] = v }
        }
        try WeightIO.apply(all, to: module, component: component)
    }

    public func prompt(_ r: SongRequest) -> HeartMuLaPrompt {
        PromptBuilder.build(tags: r.tags, lyrics: r.lyrics, tokenizer: tokenizer, config: genConfig)
    }

    /// The backbone's context (torchtune `max_seq_len` 8 192): the prompt plus every frame must fit.
    public static let maxPositions = 8192

    /// `progress` sees every LM frame, Euler step and SQ-Codec window; `isCancelled` is probed at each of those seams and
    /// ends the run with `HeartMuLaError.cancelled`.
    public func generate(_ r: SongRequest, progress: ((HeartMuLaProgress) -> Void)? = nil,
                         isCancelled: (() -> Bool)? = nil) throws -> SongResult {
        let p = prompt(r)
        guard p.length + r.maxFrames <= Self.maxPositions else {
            throw HeartMuLaError.invalidInput("the prompt (\(p.length) positions) plus \(r.maxFrames) frames exceeds the backbone's \(Self.maxPositions) positions — shorten the lyrics or the cap")
        }
        let noise = MLXNoise(seed: r.seed)
        let maxMs = r.maxMs
        let lmFrames = maxMs / 80
        let gen = try Rollout.generate(model: lm, prompt: p, maxMs: maxMs, sampling: r.sampling, noise: noise,
                                       audioEOS: genConfig.audio_eos_id, stopOnEndToken: r.stopOnEndToken,
                                       progress: { i in
                                           progress?(.generate(frame: i, maxFrames: lmFrames))
                                           return !(isCancelled?() ?? false)
                                       })
        Memory.clearCache()
        if isCancelled?() ?? false { throw HeartMuLaError.cancelled("cancelled before the codec") }
        let dec = try decode(codes: gen.codes, seed: r.seed, steps: r.codecSteps, guidance: r.codecGuidance,
                             progress: { s in progress?(s); return !(isCancelled?() ?? false) })
        let (det, audio, peak, gain) = (dec.result, dec.audio, dec.rawPeak, dec.gain)
        return SongResult(audio: audio, codes: gen.codes, sampleRate: 48_000, frames: gen.frames, endedOnEndToken: gen.endedOnEndToken,
                          effectiveSeconds: Double(audio.dim(1)) / 48_000, rawPeak: peak, limiterGain: gain,
                          promptTokens: p.length, lmSeconds: gen.seconds, prefillSeconds: gen.prefillSeconds,
                          ditSeconds: det.windowSeconds, sqSeconds: det.sqSeconds)
    }

    /// The codec stage alone: codes (8, F) → limited audio. Its noise comes from MLX's generator re-seeded with
    /// `codecSeed(seed)`, independent of the LM's stream, so saved codes decode to the same audio again (the V2b `c16` tier).
    public func decode(codes: MLXArray, seed: UInt64, steps: Int = 10, guidance: Float = 1.25,
                       progress: ((HeartMuLaProgress) -> Bool)? = nil) throws
        -> (result: DetokenizeResult, audio: MLXArray, rawPeak: Float, gain: Float) {
        MLXRandom.seed(Self.codecSeed(seed))
        let det = try Detokenizer.run(codec: codec, codes: codes, steps: steps, guidance: guidance,
                                      noise: { _, shape in MLXRandom.normal(shape, dtype: .float32) }, progress: progress)
        let (audio, peak, gain) = Limiter.apply(det.audio)
        eval(audio)
        return (det, audio, peak, gain)
    }

    public static func codecSeed(_ seed: UInt64) -> UInt64 { seed ^ 0x4845_4152_5443_4F44 }   // "HEARTCOD"
}

/// mlx-lm's bundle contract for a pre-quantised LM: `"quantization": {"group_size", "bits", "mode"}` in config.json and
/// `weight` (packed uint32) + `scales` + `biases` per quantised Linear. HeartMuLa's tiers quantise exactly the Linears inside
/// `backbone.layers.*` and `decoder.layers.*`; embeddings, `muq_linear`, `projection`, `codebook0_head`, `audio_head` and the
/// norms stay bf16. `apply` builds the `QuantizedLinear`s BEFORE the bundle's tensors are loaded (no quantise-at-load).
public struct LMQuantization: Sendable, Equatable {
    public let groupSize: Int
    public let bits: Int

    public init(groupSize: Int = 64, bits: Int) { self.groupSize = groupSize; self.bits = bits }

    public static func from(config url: URL) throws -> LMQuantization? {
        guard let q = try ConfigIO.json(url)["quantization"] as? [String: Any] else { return nil }
        guard let g = q["group_size"] as? Int, let b = q["bits"] as? Int, (q["mode"] as? String ?? "affine") == "affine" else {
            throw HeartMuLaError.badConfig("unsupported quantization block \(q)")
        }
        return LMQuantization(groupSize: g, bits: b)
    }

    public static func selects(_ path: String, _ m: Module) -> Bool {
        m is Linear && (path.hasPrefix("backbone.layers.") || path.hasPrefix("decoder.layers."))
    }

    public func apply(to model: Module) {
        quantize(model: model, groupSize: groupSize, bits: bits, mode: .affine, filter: Self.selects)
    }

    public var json: [String: Any] { ["group_size": groupSize, "bits": bits, "mode": "affine"] }
}
