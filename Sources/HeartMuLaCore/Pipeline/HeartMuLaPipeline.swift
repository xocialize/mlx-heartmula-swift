// HeartMuLaPipeline.swift — `HeartMuLaGenPipeline` end to end: preprocess → _forward (the AR rollout) → HeartCodec.detokenize →
// the output limiter. Loads the two bundles the converter writes (`HeartMuLa-oss-3B-bf16`, `HeartCodec-oss-{fp32,fp16}`)
// through the key contract (WeightIO). The limiter is the port's addition (V0 outputs peaked at 1.03–1.37 on 8 of 12 songs):
// one gain `min(1, 0.891 / peak)` over the whole song — a −1 dBFS ceiling, no compressor, no per-sample clipping.
import Foundation
import MLX
import MLXNN
import MLXRandom

public struct MusicRequest: Sendable {
    public var tags: String
    public var lyrics: String
    public var maxSeconds: Double
    public var seed: UInt64
    public var sampling: SamplingConfig
    public var codecSteps: Int
    public var codecGuidance: Float
    public init(tags: String, lyrics: String, maxSeconds: Double = 120, seed: UInt64 = 0, sampling: SamplingConfig = SamplingConfig(),
                codecSteps: Int = 10, codecGuidance: Float = 1.25) {
        self.tags = tags; self.lyrics = lyrics; self.maxSeconds = maxSeconds; self.seed = seed; self.sampling = sampling
        self.codecSteps = codecSteps; self.codecGuidance = codecGuidance
    }
}

public struct MusicResult {
    public let audio: MLXArray          // (2, samples) fp32, limited
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

    /// `lmDir` = the LM bundle (weights, config.json, tokenizer.json, gen_config.json); `codecDir` = a codec bundle.
    /// `.float32` casts every tensor to fp32 (the exact lane); `.shipped` keeps the bundle dtypes.
    public init(lmDir: URL, codecDir: URL, dtype: WeightDType = .shipped) throws {
        let lc = try ConfigIO.load(HeartMuLaConfig.self, from: lmDir.appendingPathComponent("config.json"))
        let cc = try ConfigIO.load(CodecConfig.self, from: codecDir.appendingPathComponent("config.json"))
        tokenizer = try BPETokenizer(file: lmDir.appendingPathComponent("tokenizer.json"))
        genConfig = (try? ConfigIO.load(GenConfig.self, from: lmDir.appendingPathComponent("gen_config.json"))) ?? GenConfig()
        lm = try HeartMuLaModel(lc)
        codec = HeartCodecModel(cc)
        let cast: DType? = dtype == .float32 ? .float32 : nil
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

    public func prompt(_ r: MusicRequest) -> HeartMuLaPrompt {
        PromptBuilder.build(tags: r.tags, lyrics: r.lyrics, tokenizer: tokenizer, config: genConfig)
    }

    public func generate(_ r: MusicRequest, progress: ((String, Int, Int) -> Bool)? = nil) throws -> MusicResult {
        let p = prompt(r)
        let noise = MLXNoise(seed: r.seed)
        let maxMs = Int(r.maxSeconds * 1000)
        let gen = try Rollout.generate(model: lm, prompt: p, maxMs: maxMs, sampling: r.sampling, noise: noise,
                                       audioEOS: genConfig.audio_eos_id,
                                       progress: progress.map { cb in { i in cb("lm", i, maxMs / 80) } })
        Memory.clearCache()
        let det = try Detokenizer.run(codec: codec, codes: gen.codes, steps: r.codecSteps, guidance: r.codecGuidance,
                                      noise: { _, shape in MLXRandom.normal(shape, dtype: .float32) },
                                      progress: progress.map { cb in { w, n in cb("codec", w, n) } })
        let (audio, peak, gain) = Limiter.apply(det.audio)
        eval(audio)
        return MusicResult(audio: audio, sampleRate: 48_000, frames: gen.frames, endedOnEndToken: gen.endedOnEndToken,
                           effectiveSeconds: Double(audio.dim(1)) / 48_000, rawPeak: peak, limiterGain: gain,
                           promptTokens: p.length, lmSeconds: gen.seconds, prefillSeconds: gen.prefillSeconds,
                           ditSeconds: det.windowSeconds, sqSeconds: det.sqSeconds)
    }
}
