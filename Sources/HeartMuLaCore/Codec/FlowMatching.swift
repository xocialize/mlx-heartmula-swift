// FlowMatching.swift — heartcodec/models/flow_matching.py `FlowMatching.inference_codes` + `solve_euler`, and the RVQ lookup
// (`vector_quantize_pytorch.ResidualVQ.get_output_from_indices`: Σ_q codebook_q[code_q] (32-d) → `project_out` → 512).
//   condition  = nearest ×2 (12.5 → 25 Hz) of `cond_feature_emb(rvq(codes))`; every frame of a window is inside `latent_length`
//                (mask 2) or the in-context head (mask 1), so `zero_cond_embedding1` never reaches the estimator in a 29.76 s
//                window — kept for the key contract and applied by the same mask rule.
//   in-context = `true_latents · [mask == 1]`, its length = the first `incontext_length` frames.
//   solve      = 10 Euler steps on t_span = linspace(0, 1, 11): before each step the in-context head of x is reset to
//                (1 − (1 − 1e-6)·t)·noise + t·incontext_x; the estimator runs the CFG pair [x;x], [ic;ic], [0;mu] at t, t;
//                v = v_uncond + g·(v_cond − v_uncond); x += dt·v; t += dt; dt = t_span[step+1] − t (fp32, as torch). After the
//                solve the head is overwritten with incontext_x.
import Foundation
import MLX
import MLXNN

public final class ResidualVQ: Module {
    @ModuleInfo(key: "codebooks") var codebooks: [Embedding]
    @ModuleInfo(key: "project_out") var projectOut: Linear
    init(_ c: CodecConfig) {
        _codebooks.wrappedValue = (0 ..< c.num_quantizers).map { _ in Embedding(embeddingCount: c.codebook_size, dimensions: c.codebook_dim) }
        _projectOut.wrappedValue = Linear(c.codebook_dim, c.dim)
    }
    /// codes (B, 8, T) → (B, T, 512)
    public func callAsFunction(_ codes: MLXArray) -> MLXArray {
        var acc: MLXArray? = nil
        for (q, cb) in codebooks.enumerated() {
            let e = cb(codes[0..., q, 0...].asType(.int32))
            acc = acc.map { $0 + e } ?? e
        }
        return projectOut(acc!)
    }
}

public final class FlowMatching: Module {
    @ModuleInfo(key: "vq_embed") public var vqEmbed: ResidualVQ
    @ModuleInfo(key: "cond_feature_emb") var condFeatureEmb: Linear
    @ParameterInfo(key: "zero_cond_embedding1") var zeroCond: MLXArray
    @ModuleInfo(key: "estimator") public var estimator: FlowTransformer
    public let latentDim: Int
    /// The estimator's weight dtype (fp32 or fp16 by codec tier).
    public var dtype: DType { condFeatureEmb.weight.dtype }

    init(_ c: CodecConfig) {
        _vqEmbed.wrappedValue = ResidualVQ(c)
        _condFeatureEmb.wrappedValue = Linear(c.dim, c.dim)
        _zeroCond.wrappedValue = MLXArray.zeros([c.dim])
        _estimator.wrappedValue = FlowTransformer(c)
        latentDim = c.out_channels
    }

    /// The window's condition `mu` (B, 2T, 512) from codes (B, 8, T), masked by `latent_length` (frames ≥ it take zero_cond).
    public func condition(_ codes: MLXArray, latentLength: Int) -> MLXArray {
        let q = condFeatureEmb(vqEmbed(codes))                                   // (B, T, 512)
        let (b, t, d) = (q.dim(0), q.dim(1), q.dim(2))
        let up = broadcast(q.expandedDimensions(axis: 2), to: [b, t, 2, d]).reshaped(b, 2 * t, d)
        let n = 2 * t
        guard latentLength < n else { return up }
        let keep = (MLXArray(Int32(0) ..< Int32(n)) .< Int32(latentLength)).reshaped(1, n, 1)
        return which(keep, up, broadcast(zeroCond.reshaped(1, 1, d), to: up.shape))
    }

    /// `inference_codes` (scenario "other_seg") with the noise supplied: codes (B, 8, T), trueLatents (B, 2T, 256),
    /// noise (B, 2T, 256) → latents (B, 2T, 256).
    public func inferenceCodes(_ codes: MLXArray, trueLatents: MLXArray, latentLength: Int, incontextLength: Int,
                               noise: MLXArray, guidance: Float, steps: Int, onStep: ((Int) -> Bool)? = nil) throws -> MLXArray {
        let mu = condition(codes, latentLength: latentLength)
        let n = mu.dim(1)
        let isCtx = (MLXArray(Int32(0) ..< Int32(n)) .< Int32(min(incontextLength, latentLength))).reshaped(1, n, 1)
        let ictx = trueLatents * isCtx.asType(trueLatents.dtype)
        var x = try solve(noise: noise, incontext: ictx, incontextLength: incontextLength, mu: mu, guidance: guidance, steps: steps,
                          onStep: onStep)
        if incontextLength > 0 {
            x = concatenated([ictx[0..., ..<incontextLength, 0...], x[0..., incontextLength..., 0...]], axis: 1)
        }
        return x
    }

    /// `solve_euler` with torch's fp32 time arithmetic. `onStep(step)` runs before each Euler step (1-based) and may cancel by
    /// returning false — a probe only, the arithmetic is unchanged.
    public func solve(noise: MLXArray, incontext: MLXArray, incontextLength L: Int, mu: MLXArray, guidance: Float, steps: Int,
                      onStep: ((Int) -> Bool)? = nil) throws -> MLXArray {
        let span: [Float] = (0 ... steps).map { Float($0) / Float(steps) }       // torch.linspace(0, 1, steps + 1) in fp32
        var t = span[0]
        var dt = span[1] - span[0]
        var x = noise
        let b = x.dim(0)
        for step in 1 ... steps {
            if let onStep, !onStep(step) { throw HeartMuLaError.cancelled("flow matching cancelled at step \(step)") }
            if L > 0 {
                let head = (1 - (1 - Float(1e-6)) * t) * noise[0..., ..<L, 0...] + t * incontext[0..., ..<L, 0...]
                x = concatenated([head, x[0..., L..., 0...]], axis: 1)
            }
            let v: MLXArray
            if guidance > 1 {
                let inp = concatenated([concatenated([x, x], axis: 0), concatenated([incontext, incontext], axis: 0),
                                        concatenated([MLXArray.zeros(like: mu), mu], axis: 0)], axis: 2)
                let out = estimator(inp, timestep: MLXArray([t, t]).asType(x.dtype))
                let vu = out[0 ..< b], vc = out[b ..< (2 * b)]
                v = vu + guidance * (vc - vu)
            } else {
                v = estimator(concatenated([x, incontext, mu], axis: 2), timestep: MLXArray([t]).asType(x.dtype))
            }
            x = x + dt * v
            t = t + dt
            if step < steps { dt = span[step + 1] - t }
            eval(x)
        }
        return x
    }
}
