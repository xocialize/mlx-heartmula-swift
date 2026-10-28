// Generator.swift — heartlib's `sample_topk` + `HeartMuLa.generate_frame` + `HeartMuLaGenPipeline._forward`.
//   • sample_topk: logits / temperature; entries below the k-th largest → −∞; log_softmax; softmax; then
//     `_multinomial_sample_one_no_sync` = argmax(probs / q) with q ~ Exp(1). The exponential draw is injectable (`NoiseSource`):
//     the gates feed torch's recorded q, a render draws q = −log(U) from MLX's generator (a different stream from torch's, so
//     a seed does not reproduce a torch take).
//   • CFG on both heads: guided = uncond + (cond − uncond) · cfg, one sample repeated to both rows.
//   • A frame: backbone step (prompt prefill, then one position per frame at batch 2) → c0 from `codebook0_head`; the local
//     decoder restarts its cache every frame: [last_h ; embed_0(c0)] at positions 0, 1, then embed_{i−1}(c_{i−1}) at position i,
//     `audio_head[i − 1]` → c_i.
//   • _forward: the first frame from the prompt, then up to `max_ms / 80` more; a frame whose conditional-row codes contain any
//     id ≥ audio_eos_id (8193) ends the song and is not kept.
import Foundation
import MLX
import MLXRandom

public protocol NoiseSource: AnyObject {
    /// q ~ Exp(1) shaped like `like` (B, V), fp32.
    func exponential(like: MLXArray) -> MLXArray
}

/// MLX's generator: q = −log(U), U ∈ (0, 1].
public final class MLXNoise: NoiseSource {
    public init(seed: UInt64) { MLXRandom.seed(seed) }
    public func exponential(like: MLXArray) -> MLXArray {
        let u = MLXRandom.uniform(low: Float.leastNormalMagnitude, high: 1, like.shape, dtype: .float32)
        return -log(u)
    }
}

/// A fixed sequence of recorded draws (the gates' `F_q`), consumed in order.
public final class ReplayNoise: NoiseSource {
    private var queue: [MLXArray]
    public init(_ draws: [MLXArray]) { queue = draws }
    public func exponential(like: MLXArray) -> MLXArray {
        precondition(!queue.isEmpty, "ReplayNoise exhausted")
        return queue.removeFirst().reshaped(like.shape).asType(.float32)
    }
}

public struct SamplingConfig: Sendable {
    public var cfgScale: Float = 1.5
    public var topK: Int = 50
    public var temperature: Float = 1.0
    public init(cfgScale: Float = 1.5, topK: Int = 50, temperature: Float = 1.0) {
        self.cfgScale = cfgScale; self.topK = topK; self.temperature = temperature
    }
}

public enum Sampler {
    /// Guided logits for a CFG pair (2, V) → (1, V); a single row passes through.
    public static func guided(_ logits: MLXArray, cfg: Float) -> MLXArray {
        guard logits.dim(0) == 2, cfg > 1 else { return logits }
        let cond = logits[0 ..< 1], uncond = logits[1 ..< 2]
        return uncond + (cond - uncond) * cfg
    }

    /// `sample_topk` (B, V) → (B,) int32, probabilities computed in fp32.
    public static func sampleTopK(_ logits: MLXArray, topK: Int, temperature: Float, noise: NoiseSource) -> MLXArray {
        let l = logits.asType(.float32) / temperature
        let v = l.dim(-1)
        let kth = sorted(l, axis: -1)[.ellipsis, (v - topK) ..< (v - topK + 1)]
        let filtered = which(l .< kth, MLXArray(-Float.infinity), l)
        let lsm = filtered - logSumExp(filtered, axis: -1, keepDims: true)
        let probs = softmax(lsm, axis: -1)
        let q = noise.exponential(like: probs)
        return argMax(probs / q, axis: -1).asType(.int32)
    }
}

/// Per-frame record for the gates (lane F).
public struct FrameTrace {
    public var guidedLogits: [MLXArray] = []    // 8 × (1, V)
    public var c0Logits: MLXArray?              // (2, V)
    public init() {}
}

public final class FrameGenerator {
    public let model: HeartMuLaModel
    public let sampling: SamplingConfig
    public private(set) var caches: [KVCache]
    public private(set) var position = 0

    public init(model: HeartMuLaModel, sampling: SamplingConfig) {
        self.model = model; self.sampling = sampling
        caches = model.backbone.newCaches()
    }

    private var cfgActive: Bool { sampling.cfgScale > 1 }

    /// One `generate_frame` from the backbone input `h` (B, L, D) (the prompt embedding, or a frame embedding for one step).
    /// Returns the 8 codes (8,) int32, lazily evaluated.
    public func frame(_ h: MLXArray, noise: NoiseSource, trace: UnsafeMutablePointer<FrameTrace>? = nil) -> MLXArray {
        let hid = model.backbone(h, caches: caches)
        position += h.dim(1)
        let lastH = hid[0..., (hid.dim(1) - 1) ..< hid.dim(1), 0...]                 // (B, 1, D)
        let c0l = model.c0Logits(lastH.squeezed(axis: 1))                             // (B, V)
        let g0 = Sampler.guided(c0l, cfg: sampling.cfgScale)
        var codes = [Sampler.sampleTopK(g0, topK: sampling.topK, temperature: sampling.temperature, noise: noise)]   // (1,)
        trace?.pointee.c0Logits = c0l; trace?.pointee.guidedLogits.append(g0)
        let b = h.dim(0)
        let dcaches = model.decoder.newCaches(step: 8)
        var cur = concatenated([lastH, model.embedAudio(0, broadcast(codes[0], to: [b])).expandedDimensions(axis: 1).asType(lastH.dtype)], axis: 1)
        for i in 1 ..< model.codebooks {
            let dh = model.localDecoder(cur, caches: dcaches)
            let li = model.audioLogits(dh[0..., dh.dim(1) - 1, 0...], codebook: i)        // (B, V)
            let gi = Sampler.guided(li, cfg: sampling.cfgScale)
            let ci = Sampler.sampleTopK(gi, topK: sampling.topK, temperature: sampling.temperature, noise: noise)
            trace?.pointee.guidedLogits.append(gi)
            codes.append(ci)
            cur = model.embedAudio(i, broadcast(ci, to: [b])).expandedDimensions(axis: 1).asType(lastH.dtype)
        }
        return concatenated(codes, axis: 0)                                            // (8,)
    }

    /// The backbone input for the next step from the previous frame's codes: (8,) → (B, 1, D).
    public func nextInput(_ codes: MLXArray) -> MLXArray {
        let e = model.frameEmbedding(codes.reshaped(1, model.codebooks))               // (1, D)
        let b = cfgActive ? 2 : 1
        return broadcast(e.expandedDimensions(axis: 0), to: [b, 1, e.dim(-1)]).asType(model.dtype)
    }
}

public struct GenerationResult {
    public let codes: MLXArray            // (8, F) int32
    public let frames: Int
    public let endedOnEndToken: Bool
    public let seconds: Double
    public let prefillSeconds: Double
}

public enum Rollout {
    /// `_forward`: prompt → first frame; then up to `maxMs / 80` frames; stop on the end token. `progress(frame)` may cancel.
    public static func generate(model: HeartMuLaModel, prompt: HeartMuLaPrompt, maxMs: Int, sampling: SamplingConfig,
                                noise: NoiseSource, audioEOS: Int = 8193,
                                progress: ((Int) -> Bool)? = nil) throws -> GenerationResult {
        let gen = FrameGenerator(model: model, sampling: sampling)
        let t0 = Date()
        var h = model.promptEmbedding(prompt).asType(model.dtype)
        if sampling.cfgScale <= 1 { h = h[0 ..< 1] }
        var frames: [MLXArray] = []
        var code = gen.frame(h, noise: noise)
        eval(code)
        let prefill = Date().timeIntervalSince(t0)
        frames.append(code)
        var ended = false
        for i in 0 ..< (maxMs / 80) {
            if let progress, !progress(i) { throw HeartMuLaError.cancelled("generation cancelled at frame \(i)") }
            code = gen.frame(gen.nextInput(code), noise: noise)
            eval(code)
            if code.max().item(Int32.self) >= Int32(audioEOS) { ended = true; break }
            frames.append(code)
            if i % 64 == 63 { Memory.clearCache() }
        }
        let codes = stacked(frames, axis: 1)                                           // (8, F)
        eval(codes)
        return GenerationResult(codes: codes, frames: frames.count, endedOnEndToken: ended,
                                seconds: Date().timeIntervalSince(t0), prefillSeconds: prefill)
    }
}
