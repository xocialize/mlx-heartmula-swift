// Detokenizer.swift — heartcodec/models/decoder.py `HeartCodecDecoder.detokenize` (duration 29.76 s, 10 steps, guidance 1.25):
//   frames:  min = int(29.76·12.5) = 372, hop = 372 // 93 · 80 = 320, overlap 52 frames = 104 latents;
//   codes shorter than 372 are doubled (cat(codes, codes)) and cut to 372; then doubled again and cut so that
//   (len − 52) is a multiple of 320; windows start at 0, 320, … ≤ len − 320;
//   window 0 runs with no in-context head; window k > 0 carries the last 104 latents of window k − 1 as its in-context head;
//   each window's latents (744 × 256) split into 2 × 128 channel halves (stereo as batch) → SQ-Codec → 1 428 480 samples;
//   audio windows overlap by 1 428 480 − 1 228 800 = 199 680 samples with a linear crossfade (numpy float64 `linspace`, as
//   upstream computes it on the CPU in float64 and stores fp32); the song is cut to int(F / 12.5 · 48 000) samples.
// The per-window noise is supplied (`noise(window, shape)`) — the gates pass torch's recorded draws, a render MLX's normal.
import Foundation
import MLX
import MLXRandom

public struct CodecWindowing: Sendable {
    public static let duration = 29.76
    public static let minFrames = Int(duration * 12.5)                 // 372
    public static let hopFrames = minFrames / 93 * 80                  // 320
    public static let overlapFrames = minFrames - hopFrames            // 52
    public static let overlapLatents = overlapFrames * 2               // 104
    public static let latentLength = Int(duration * 25)                // 744
    public static let windowSamples = Int(duration * 48_000)           // 1 428 480
    public static let hopSamples = windowSamples / 93 * 80             // 1 228 800
    public static let overlapSamples = windowSamples - hopSamples      // 199 680

    /// The upstream code-repeat rule: indices into the original F frames for the padded sequence.
    public static func paddedIndices(frames f: Int) -> [Int] {
        var idx = Array(0 ..< f)
        if idx.count < minFrames {
            while idx.count < minFrames { idx += idx }
            idx = Array(idx.prefix(minFrames))
        }
        let n = idx.count
        if (n - overlapFrames) % hopFrames > 0 {
            let target = Int((Double(n - overlapFrames) / Double(hopFrames)).rounded(.up)) * hopFrames + overlapFrames
            while idx.count < target { idx += idx }
            idx = Array(idx.prefix(target))
        }
        return idx
    }

    public static func windowStarts(paddedLength n: Int) -> [Int] { Array(stride(from: 0, through: n - hopFrames, by: hopFrames)) }

    public static func targetSamples(frames f: Int) -> Int { Int(Double(f) / 12.5 * 48_000) }
}

public struct DetokenizeResult {
    public let audio: MLXArray            // (2, samples) fp32
    public let windowLatents: [MLXArray]  // per window (1, 744, 256)
    public let windowSeconds: [Double]
    public let sqSeconds: Double
}

/// Where a song is, for a host's progress bar and its cancellation probe.
public enum HeartMuLaProgress: Sendable, Equatable {
    /// LM frame `frame` (0-based) of at most `maxFrames` (the cap at 12.5 Hz).
    case generate(frame: Int, maxFrames: Int)
    /// Euler step `step` (1-based) of `steps`, in codec window `window` (0-based) of `windows`.
    case denoise(window: Int, windows: Int, step: Int, steps: Int)
    /// SQ-Codec window `window` (0-based) of `windows`.
    case decode(window: Int, windows: Int)
}

public enum Detokenizer {
    /// `progress` sees every Euler step of every window (`.denoise`) and every SQ-Codec window (`.decode`); returning false
    /// cancels at that seam.
    public static func run(codec: HeartCodecModel, codes: MLXArray, steps: Int = 10, guidance: Float = 1.25,
                           noise: (Int, [Int]) -> MLXArray, progress: ((HeartMuLaProgress) -> Bool)? = nil) throws -> DetokenizeResult {
        let f = codes.dim(1)
        let idx = CodecWindowing.paddedIndices(frames: f)
        let padded = take(codes, MLXArray(idx.map { Int32($0) }), axis: 1).expandedDimensions(axis: 0)   // (1, 8, N)
        let starts = CodecWindowing.windowStarts(paddedLength: idx.count)
        let fm = codec.flowMatching
        let dtype = fm.dtype
        let L = CodecWindowing.latentLength
        var latents: [MLXArray] = []
        var times: [Double] = []
        for (w, s) in starts.enumerated() {
            let t0 = Date()
            let c = padded[0..., 0..., s ..< (s + CodecWindowing.minFrames)]
            let z = noise(w, [1, L, fm.latentDim]).asType(dtype)
            let onStep = progress.map { cb in { (step: Int) in cb(.denoise(window: w, windows: starts.count, step: step, steps: steps)) } }
            let out: MLXArray
            if w == 0 {
                out = try fm.inferenceCodes(c, trueLatents: MLXArray.zeros([1, L, fm.latentDim], dtype: dtype), latentLength: L,
                                            incontextLength: 0, noise: z, guidance: guidance, steps: steps, onStep: onStep)
            } else {
                let carry = latents[w - 1][0..., (L - CodecWindowing.overlapLatents)..., 0...]
                let tl = concatenated([carry, MLXArray.zeros([1, L - CodecWindowing.overlapLatents, fm.latentDim], dtype: dtype)], axis: 1)
                out = try fm.inferenceCodes(c, trueLatents: tl, latentLength: L, incontextLength: CodecWindowing.overlapLatents,
                                            noise: z, guidance: guidance, steps: steps, onStep: onStep)
            }
            eval(out)
            latents.append(out)
            times.append(Date().timeIntervalSince(t0))
        }
        let t1 = Date()
        let audio = try assemble(codec: codec, latents: latents, frames: f,
                                 onWindow: progress.map { cb in { w in cb(.decode(window: w, windows: latents.count)) } })
        return DetokenizeResult(audio: audio, windowLatents: latents, windowSeconds: times, sqSeconds: Date().timeIntervalSince(t1))
    }

    /// SQ-Codec per window, the float64 linear crossfade, the cut to int(F / 12.5 · 48 000) samples → (2, samples) fp32.
    public static func assemble(codec: HeartCodecModel, latents: [MLXArray], frames f: Int,
                                onWindow: ((Int) -> Bool)? = nil) throws -> MLXArray {
        var output: [[Float]] = [[], []]
        let ov = CodecWindowing.overlapSamples
        // numpy linspace(0, 1, ov) in float64: start + i·step, last = stop
        let step = 1.0 / Double(ov - 1)
        let fadeIn: [Double] = (0 ..< ov).map { $0 == ov - 1 ? 1.0 : Double($0) * step }
        for (w, lat) in latents.enumerated() {
            if let onWindow, !onWindow(w) { throw HeartMuLaError.cancelled("SQ-Codec decode cancelled at window \(w)") }
            let wav = decodeWindow(codec: codec, latents: lat)                                   // (2, windowSamples)
            eval(wav)
            let a = wav.asType(.float32)
            for ch in 0 ..< 2 {
                let cur = a[ch].asArray(Float.self)
                if output[ch].isEmpty { output[ch] = cur; continue }
                let base = output[ch].count - ov
                for i in 0 ..< ov {
                    output[ch][base + i] = Float(Double(output[ch][base + i]) * (1 - fadeIn[i]) + Double(cur[i]) * fadeIn[i])
                }
                output[ch] += cur[ov...]
            }
            Memory.clearCache()
        }
        let n = min(CodecWindowing.targetSamples(frames: f), output[0].count)
        return MLXArray(Array(output[0].prefix(n)) + Array(output[1].prefix(n)), [2, n])
    }

    /// One window's latents (1, 744, 256) → stereo (2, 1 428 480): channel halves as batch, SQ-Codec, cut to the window length.
    public static func decodeWindow(codec: HeartCodecModel, latents: MLXArray) -> MLXArray {
        let (t, d) = (latents.dim(1), latents.dim(2))
        let x = latents.reshaped(t, 2, d / 2).transposed(1, 0, 2)                                // (2, T, 128)
        let y = codec.scalarModel.decode(x)
        return y[0..., ..<min(y.dim(1), CodecWindowing.windowSamples)]
    }
}
