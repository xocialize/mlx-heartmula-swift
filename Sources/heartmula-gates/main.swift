// heartmula-gates — V1 / V2 lanes for the HeartMuLa port (WIP/heartmula-port/MEASUREMENTS.md, rules pre-registered 2026-10-28).
//   contract  <lm> <codec>…                       S0: weight-free module paths == bundle keys (every tier)
//   convcheck <codec-fp32> <refs>                 V1 (b): MLX convs on the converted layouts vs torch (CPU fp32)
//   tokenizer <lm> <goldens/tokenizer>            lane T on the 12 V0 rows
//   core      <lm> <codec> <goldens/case> [--dtype float32|shipped]   lanes S E P L F N W Q D
//   render    <lm> <codec> <prompts.json> <id> <out.wav> [--max 120] [--seed N] [--cfg 1.5]
// Run the fp32 lanes with MLX_ENABLE_TF32=0. relL2 = ‖Δ‖₂/‖ref‖₂, SNR dB = −20·log₁₀ relL2.
import Foundation
import HeartMuLaCore
import MLX
import MLXNN
import MLXRandom

setvbuf(stdout, nil, _IOLBF, 0)
var args = Array(CommandLine.arguments.dropFirst())
func opt(_ name: String, _ def: String) -> String {
    if let i = args.firstIndex(of: name), i + 1 < args.count { let v = args[i + 1]; args.removeSubrange(i ... i + 1); return v }
    return def
}
let dtypeName = opt("--dtype", "float32")
let maxSeconds = Double(opt("--max", "120"))!
let seed = UInt64(opt("--seed", "1004"))!
let cfgScale = Float(opt("--cfg", "1.5"))!
let cacheGB = Double(opt("--cache-gb", "2"))!
Memory.cacheLimit = Int(cacheGB * Double(1 << 30))
guard let cmd = args.first else { print("usage: see header"); exit(2) }
let rest = Array(args.dropFirst())
func url(_ s: String) -> URL { URL(fileURLWithPath: s) }
func npy(_ d: URL, _ n: String) throws -> MLXArray { try NPY.load(d.appendingPathComponent(n + ".npy")) }
func ts() -> String { let f = DateFormatter(); f.dateFormat = "HH:mm:ss"; return f.string(from: Date()) }
func log(_ s: String) { print("\(ts()) \(s)") }

func relL2(_ a: MLXArray, _ b: MLXArray) -> Double {
    let x = a.asType(.float32), y = b.asType(.float32)
    precondition(x.shape == y.shape, "shape \(x.shape) vs \(y.shape)")
    let num = sqrt(sum((x - y) * (x - y))).item(Float.self), den = sqrt(sum(y * y)).item(Float.self)
    return Double(num) / Double(max(den, 1e-30))
}
func snr(_ r: Double) -> Double { -20 * log10(max(r, 1e-30)) }
func fmt(_ v: Double) -> String { String(format: "%.2e", v) }

var results: [(String, Bool, String)] = []
func record(_ lane: String, _ pass: Bool, _ detail: String) {
    results.append((lane, pass, detail)); log("\(pass ? "PASS" : "FAIL") \(lane): \(detail)")
}

/// Argmax agreement with the pre-registered near-tie clause: positions whose golden top-2 margin < 1e-3 are excluded.
func argmaxAgreement(_ a: MLXArray, _ ref: MLXArray) -> (agree: Int, total: Int, ties: Int, tieDisagree: Int) {
    let v = ref.dim(-1)
    let ra = ref.reshaped(-1, v).asType(.float32), aa = a.reshaped(-1, v).asType(.float32)
    let am = argMax(aa, axis: -1).asArray(Int32.self), rm = argMax(ra, axis: -1).asArray(Int32.self)
    let srt = sorted(ra, axis: -1)
    let margin = (srt[0..., v - 1] - srt[0..., v - 2]).asArray(Float.self)
    var agree = 0, total = 0, ties = 0, tieDis = 0
    for i in 0 ..< am.count {
        if margin[i] < 1e-3 { ties += 1; if am[i] != rm[i] { tieDis += 1 }; continue }
        total += 1; if am[i] == rm[i] { agree += 1 }
    }
    return (agree, total, ties, tieDis)
}

func loadPipeline(_ lm: String, _ codec: String, _ dt: WeightDType) throws -> HeartMuLaPipeline {
    let t = Date()
    let p = try HeartMuLaPipeline(lmDir: url(lm), codecDir: url(codec), dtype: dt)
    log("loaded \(url(lm).lastPathComponent) + \(url(codec).lastPathComponent) as \(dt.rawValue) in \(String(format: "%.1f", Date().timeIntervalSince(t)))s; "
        + "phys \(String(format: "%.2f", Double(Footprint.physBytes()) / 1e9)) GB; MLX active \(String(format: "%.2f", Double(Memory.activeMemory) / 1e9)) GB")
    return p
}

func finish() -> Never {
    let failed = results.filter { !$0.1 }
    print("SUMMARY \(results.count - failed.count)/\(results.count) PASS" + (failed.isEmpty ? "" : " — FAIL: \(failed.map(\.0).joined(separator: ", "))"))
    exit(failed.isEmpty ? 0 : 1)
}

// MARK: - contract (S0)

func safetensorKeys(_ file: URL) throws -> [String] {
    let h = try FileHandle(forReadingFrom: file)
    defer { try? h.close() }
    let lenData = h.readData(ofLength: 8)
    let n = lenData.withUnsafeBytes { $0.load(as: UInt64.self) }
    let header = h.readData(ofLength: Int(n))
    let obj = try JSONSerialization.jsonObject(with: header) as! [String: Any]
    return obj.keys.filter { $0 != "__metadata__" }
}

func bundleKeys(_ dir: URL) throws -> Set<String> {
    var keys = Set<String>()
    for s in try WeightIO.shards(in: dir) { keys.formUnion(try safetensorKeys(s)) }
    return keys
}

func contract() throws {
    guard rest.count >= 2 else { print("contract <lm> <codec>…"); exit(2) }
    let lc = try ConfigIO.load(HeartMuLaConfig.self, from: url(rest[0]).appendingPathComponent("config.json"))
    let lm = try HeartMuLaModel(lc)
    let lmPaths = Set(lm.parameters().flattened().map(\.0))
    let lk = try bundleKeys(url(rest[0]))
    record("S0 \(url(rest[0]).lastPathComponent)", lmPaths == lk,
           "\(lmPaths.count) module paths vs \(lk.count) keys; missing \(lmPaths.subtracting(lk).sorted().prefix(4)) extra \(lk.subtracting(lmPaths).sorted().prefix(4))")
    for c in rest.dropFirst() {
        let cc = try ConfigIO.load(CodecConfig.self, from: url(c).appendingPathComponent("config.json"))
        let codec = HeartCodecModel(cc)
        let cp = Set(codec.parameters().flattened().map(\.0))
        let ck = try bundleKeys(url(c))
        record("S0 \(url(c).lastPathComponent)", cp == ck,
               "\(cp.count) module paths vs \(ck.count) keys; missing \(cp.subtracting(ck).sorted().prefix(4)) extra \(ck.subtracting(cp).sorted().prefix(4))")
    }
}

// MARK: - convcheck (V1 b)

func convcheck() throws {
    guard rest.count == 2 else { print("convcheck <codec-fp32> <refs>"); exit(2) }
    try Device.withDefaultDevice(.cpu) {
        let cc = try ConfigIO.load(CodecConfig.self, from: url(rest[0]).appendingPathComponent("config.json"))
        let codec = HeartCodecModel(cc)
        var all = [String: MLXArray]()
        for s in try WeightIO.shards(in: url(rest[0])) { for (k, v) in try WeightIO.load(s, dtype: nil) { all[k] = v } }
        try WeightIO.apply(all, to: codec, component: "codec")
        let r = url(rest[1])
        func relmax(_ a: MLXArray, _ b: MLXArray) -> Double {
            Double((abs(a - b).max() / abs(b).max()).item(Float.self))
        }
        let sq = codec.scalarModel
        // torch (B, C, T) ↔ MLX (B, T, C)
        let y0 = sq.preConv(try npy(r, "x0").transposed(0, 2, 1)).transposed(0, 2, 1)
        let y1 = sq.blocks[0].upConv(try npy(r, "x1").transposed(0, 2, 1)).transposed(0, 2, 1)
        let y2 = sq.blocks[0].convs[3].conv1(try npy(r, "x2").transposed(0, 2, 1)).transposed(0, 2, 1)
        for (n, y, ref) in [("look-ahead conv k5", y0, try npy(r, "y0")), ("causal up-conv k10 s5", y1, try npy(r, "y1")),
                            ("causal dilated conv k7 d7", y2, try npy(r, "y2"))] {
            let shapeOK = y.shape == ref.shape
            let rm = shapeOK ? relmax(y, ref) : .infinity
            record("V1b \(n)", shapeOK && rm <= 1e-5, "shape \(y.shape) vs \(ref.shape); relmax \(fmt(rm)) (≤ 1e-5)")
        }
    }
}

// MARK: - tokenizer (T)

func tokenizerLane() throws {
    guard rest.count == 2 else { print("tokenizer <lm> <goldens/tokenizer>"); exit(2) }
    let tok = try BPETokenizer(file: url(rest[0]).appendingPathComponent("tokenizer.json"))
    let root = url(rest[1])
    let rows = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { !$0.hasPrefix(".") }.sorted()
    var ok = 0
    for rid in rows {
        let d = root.appendingPathComponent(rid)
        let meta = try ConfigIO.json(d.appendingPathComponent("meta.json"))
        let p = PromptBuilder.build(tags: meta["tags"] as! String, lyrics: meta["lyrics"] as! String, tokenizer: tok)
        let gt = try npy(d, "tokens").asArray(Int32.self), gm = try npy(d, "tokens_mask").asArray(Int32.self)
        let same = gt == p.tokens && gm == p.mask.map { $0 ? 1 : 0 } && (meta["muq_idx"] as! Int) == p.muqIndex
            && (meta["prompt_len"] as! Int) == p.length
        if same { ok += 1 } else {
            let gtext = stride(from: 8, to: gt.count, by: 9).map { gt[$0] }, stext = stride(from: 8, to: p.tokens.count, by: 9).map { p.tokens[$0] }
            let first = zip(gtext, stext).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(gtext.count, stext.count)
            log("  \(rid): golden L \(gtext.count) vs \(stext.count); first diff at \(first): \(gtext.dropFirst(first).prefix(6)) vs \(stext.dropFirst(first).prefix(6))")
        }
    }
    record("T tokenizer + preprocess", ok == rows.count, "\(ok)/\(rows.count) rows id-exact (tokens, mask, muq_idx, length)")
}

// MARK: - core lanes

func core() throws {
    guard rest.count == 3 else { print("core <lm> <codec> <goldens/case>"); exit(2) }
    let dt: WeightDType = dtypeName == "shipped" ? .shipped : .float32
    let pipe = try loadPipeline(rest[0], rest[1], dt)
    let g = url(rest[2])
    let lmDir = g.appendingPathComponent(dt == .float32 ? "fp32" : "bf16")
    let cd = g.appendingPathComponent("codec")
    let meta = try ConfigIO.json(g.appendingPathComponent("meta.json"))
    let rid = meta["id"] as! String
    let tmeta = try ConfigIO.json(g.deletingLastPathComponent().appendingPathComponent("tokenizer/\(rid)/meta.json"))
    let prompt = PromptBuilder.build(tags: tmeta["tags"] as! String, lyrics: tmeta["lyrics"] as! String, tokenizer: pipe.tokenizer)
    let lm = pipe.lm
    let codes = try npy(g, "codes")                                                     // (8, F)
    log("case \(g.lastPathComponent) (\(rid)): prompt \(prompt.length), frames \(codes.dim(1)), LM lane \(lmDir.lastPathComponent)")

    // S
    do {
        let tb = lm.backbone.theta.value, td = lm.decoder.theta.value
        let gb = try npy(g, "S_rope_theta_backbone_fp32"), gd = try npy(g, "S_rope_theta_decoder_fp32")
        let rb = Double((abs(tb - gb) / abs(gb)).max().item(Float.self)), rd = Double((abs(td - gd) / abs(gd)).max().item(Float.self))
        let temb = FlowEmbeddings.sinusoid(MLXArray([Float(0.6), Float(0.6)]))
        let re = relL2(temb, try npy(cd, "S_timestep_emb"))
        let span = (0 ... 10).map { Float($0) / 10 }, gspan = try npy(cd, "S_t_span").asArray(Float.self)
        let spanDiff = zip(span, gspan).map { abs($0 - $1) }.max()!
        let idx = CodecWindowing.paddedIndices(frames: codes.dim(1))
        let goldenWindows = (meta["codec"] as! [String: Any])["windows"] as! Int
        let goldenSamples = try npy(cd, "D_wave").dim(1)
        let ints = CodecWindowing.minFrames == 372 && CodecWindowing.hopFrames == 320 && CodecWindowing.overlapFrames == 52
            && CodecWindowing.overlapLatents == 104 && CodecWindowing.windowSamples == 1_428_480 && CodecWindowing.hopSamples == 1_228_800
            && CodecWindowing.overlapSamples == 199_680 && CodecWindowing.windowStarts(paddedLength: idx.count).count == goldenWindows
            && CodecWindowing.targetSamples(frames: codes.dim(1)) == goldenSamples
        record("S scalars", rb <= 1e-7 && rd <= 1e-7 && re <= 1e-6 && ints,
               "θ backbone rel \(fmt(rb)), decoder \(fmt(rd)) (≤ 1e-7); timestep emb relL2 \(fmt(re)) (≤ 1e-6); windowing ints \(ints) (padded \(idx.count), starts \(CodecWindowing.windowStarts(paddedLength: idx.count))); t_span max|Δ| \(fmt(Double(spanDiff))) (reported)")
    }

    // E
    let h = lm.promptEmbedding(prompt)
    let eRel = relL2(h, try npy(lmDir, "E_h"))
    record("E prompt embedding", eRel <= 1e-6, "relL2 \(fmt(eRel)) (≤ 1e-6)")

    // P
    let L = prompt.length
    let pFrames = meta["p_frames"] as! Int
    let fe = lm.frameEmbedding(codes[0..., ..<pFrames].transposed(1, 0))                 // (64, D)
    let x = concatenated([h, broadcast(fe.expandedDimensions(axis: 0), to: [2, pFrames, fe.dim(1)])], axis: 1).asType(lm.dtype)
    let xRel = relL2(x, try npy(lmDir, "P_input"))
    let hid = lm.backbone(x, caches: nil)
    let logits = lm.c0Logits(hid[0..., (L - 1)..., 0...])
    eval(hid, logits)
    let gl = try npy(lmDir, "P_logits")
    let pa = argmaxAgreement(logits, gl)
    let pl = relL2(logits, gl), ph = relL2(hid[0..., (hid.dim(1) - 16)..., 0...], try npy(lmDir, "P_hidden_tail"))
    record("P backbone", pa.agree == pa.total && pl <= 1e-3 && ph <= 1e-3,
           "argmax \(pa.agree)/\(pa.total) (near-ties excluded \(pa.ties), of which disagree \(pa.tieDisagree)); logits relL2 \(fmt(pl)); hidden tail \(fmt(ph)) (≤ 1e-3); input relL2 \(fmt(xRel))")

    // L
    var lOK = true, lDetail: [String] = []
    for j in (meta["l_frames"] as! [Int]) {
        let gin = try npy(lmDir, "L_f\(j)_input").asType(lm.dtype)
        let c = codes[0..., j]
        let emb = stacked((0 ..< 7).map { lm.embedAudio($0, broadcast(c[$0 ..< ($0 + 1)], to: [2])) }, axis: 1)
        let sin = concatenated([hid[0..., (L - 1 + j) ..< (L + j), 0...], emb.asType(lm.dtype)], axis: 1)
        let inRel = relL2(sin, gin)
        func lg(_ inp: MLXArray) -> MLXArray {
            let dh = lm.localDecoder(inp, caches: nil)
            return stacked((1 ..< 8).map { lm.audioLogits(dh[0..., $0, 0...], codebook: $0) }, axis: 1)
        }
        let gref = try npy(lmDir, "L_f\(j)_logits")
        let lgG = lg(gin), lgS = lg(sin)
        let r = relL2(lgG, gref), rc = relL2(lgS, gref)
        let a = argmaxAgreement(lgG, gref)
        lOK = lOK && r <= 1e-3 && a.agree == a.total
        lDetail.append("f\(j): \(fmt(r)) argmax \(a.agree)/\(a.total)\(a.ties > 0 ? " ties \(a.ties)" : "") [chained \(fmt(rc)), input \(fmt(inRel))]")
    }
    record("L local decoder", lOK, lDetail.joined(separator: "; ") + " (≤ 1e-3, 100 %)")

    // F
    do {
        let q = try npy(lmDir, "F_q")                                                    // (8, V)
        let replay = ReplayNoise((0 ..< 8).map { q[$0 ..< ($0 + 1)] })
        let gen = FrameGenerator(model: lm, sampling: SamplingConfig())
        var trace = FrameTrace()
        let fc = withUnsafeMutablePointer(to: &trace) { gen.frame(h.asType(lm.dtype), noise: replay, trace: $0) }
        eval(fc)
        let gcodes = try npy(lmDir, "F_codes").asArray(Int32.self), scodes = fc.asArray(Int32.self)
        let gg = try npy(lmDir, "F_guided_logits")
        let rels = (0 ..< 8).map { relL2(trace.guidedLogits[$0].reshaped(-1), gg[$0]) }
        let c0r = relL2(trace.c0Logits!, try npy(lmDir, "F_c0_logits"))
        record("F CFG frame", rels.allSatisfy { $0 <= 1e-3 } && gcodes == scodes,
               "codes \(scodes) vs golden \(gcodes); guided logits relL2 max \(fmt(rels.max()!)) (≤ 1e-3); c0 pair \(fmt(c0r))")
    }

    // N, W, Q, D — the codec lanes (fp32 codec in both dtype lanes)
    let codec = pipe.codec, fm = codec.flowMatching
    let cdt = fm.dtype
    var nDetail: [String] = [], nOK = true
    for w in 0 ..< 2 {
        let xw = try npy(cd, "N_w\(w)_x").asType(cdt), ic = try npy(cd, "W_w\(w)_incontext_x").asType(cdt), mu = try npy(cd, "W_w\(w)_mu").asType(cdt)
        let inp = concatenated([concatenated([xw, xw], axis: 0), concatenated([ic, ic], axis: 0), concatenated([MLXArray.zeros(like: mu), mu], axis: 0)], axis: 2)
        let v = fm.estimator(inp, timestep: MLXArray([Float(0.6), Float(0.6)]).asType(cdt))
        let gv = try npy(cd, "N_w\(w)_velocity")
        let ru = relL2(v[0 ..< 1], gv[0 ..< 1]), rc = relL2(v[1 ..< 2], gv[1 ..< 2])
        nOK = nOK && ru <= 1e-3 && rc <= 1e-3
        nDetail.append("w\(w) uncond \(fmt(ru)) cond \(fmt(rc))")
    }
    record("N DiT velocity t=0.6", nOK, nDetail.joined(separator: "; ") + " (≤ 1e-3)")

    var wDetail: [String] = [], wOK = true
    var swiftLat: [MLXArray] = []
    let noises = [try npy(cd, "W_w0_noise"), try npy(cd, "W_w1_noise")]
    for w in 0 ..< 2 {
        let wc = try npy(cd, "W_w\(w)_codes").expandedDimensions(axis: 0)                // (1, 8, 372)
        let mu = fm.condition(wc, latentLength: CodecWindowing.latentLength)
        let rmu = relL2(mu, try npy(cd, "W_w\(w)_mu"))
        let tl: MLXArray
        if w == 0 { tl = MLXArray.zeros([1, 744, 256], dtype: cdt) } else {
            tl = concatenated([swiftLat[0][0..., (744 - 104)..., 0...], MLXArray.zeros([1, 640, 256], dtype: cdt)], axis: 1)
        }
        let t0 = Date()
        let out = fm.inferenceCodes(wc, trueLatents: tl, latentLength: 744, incontextLength: w == 0 ? 0 : 104,
                                    noise: noises[w].asType(cdt), guidance: 1.25, steps: 10)
        eval(out)
        let secs = Date().timeIntervalSince(t0)
        swiftLat.append(out)
        let rl = relL2(out, try npy(cd, "W_w\(w)_latents"))
        wOK = wOK && rmu <= 1e-5 && rl <= 1e-3
        wDetail.append("w\(w) condition \(fmt(rmu)) latents \(fmt(rl)) (\(String(format: "%.1f", secs)) s)")
    }
    record("W window solve", wOK, wDetail.joined(separator: "; ") + " (≤ 1e-5 / ≤ 1e-3)")

    do {
        let qin = try npy(cd, "Q_input").transposed(0, 2, 1).asType(cdt)                  // (2, 744, 128)
        let t0 = Date()
        let y = codec.scalarModel.decode(qin)
        eval(y)
        let secs = Date().timeIntervalSince(t0)
        let gq = try npy(cd, "Q_wave")
        let n = y.dim(1) == gq.dim(1)
        let r = n ? relL2(y, gq) : .infinity
        record("Q SQ-Codec", n && snr(r) >= 80, "\(y.shape) vs \(gq.shape); SNR \(String(format: "%.1f", snr(r))) dB (≥ 80); \(String(format: "%.1f", secs)) s")
    }

    do {
        let dcodes = try npy(cd, "D_codes")
        let t0 = Date()
        let det = try Detokenizer.run(codec: codec, codes: dcodes, noise: { w, _ in noises[w] })
        let secs = Date().timeIntervalSince(t0)
        let gw = try npy(cd, "D_wave")
        let n = det.audio.dim(1) == gw.dim(1)
        let r = n ? relL2(det.audio, gw) : .infinity
        var detail = "\(det.audio.shape) vs \(gw.shape); SNR \(String(format: "%.1f", snr(r))) dB (≥ 80); \(String(format: "%.1f", secs)) s"
        var pass = n && snr(r) >= 80
        // Rounding near-tie clause: grid disagreements must all be golden near-ties, and the golden grid substituted at those
        // elements must reach ≥ 80 dB.
        var mism = 0, nonTie = 0
        var subst: [MLXArray] = []
        for w in 0 ..< det.windowLatents.count {
            let s = det.windowLatents[w], gl = try npy(cd, "W_w\(w)_latents").asType(s.dtype)
            let m = SQDecoder.grid(s) .!= SQDecoder.grid(gl)
            let frac = 9 * gl - floor(9 * gl)
            let tie = abs(frac - 0.5) .< 1e-3
            mism += m.asType(.int32).sum().item(Int.self)
            nonTie += (m .&& .!tie).asType(.int32).sum().item(Int.self)
            subst.append(which(m, gl, s))
        }
        detail += "; grid disagreements \(mism) (non-tie \(nonTie))"
        if !pass && n && nonTie == 0 && mism > 0 {
            let a2 = Detokenizer.assemble(codec: codec, latents: subst, frames: dcodes.dim(1))
            let r2 = relL2(a2, gw)
            detail += "; golden grid substituted at the \(mism) ties → SNR \(String(format: "%.1f", snr(r2))) dB"
            pass = snr(r2) >= 80
        }
        record("D end to end", pass, detail)
        let (lim, peak, gain) = Limiter.apply(det.audio)
        log("  limiter on D: raw peak \(String(format: "%.3f", peak)), gain \(String(format: "%.3f", gain)), limited peak \(String(format: "%.3f", abs(lim).max().item(Float.self)))")
    }
    log("phys \(String(format: "%.2f", Double(Footprint.physBytes()) / 1e9)) GB; MLX peak \(String(format: "%.2f", Double(Memory.peakMemory) / 1e9)) GB")
}

// MARK: - render

func render() throws {
    guard rest.count == 5 else { print("render <lm> <codec> <prompts.json> <id> <out.wav>"); exit(2) }
    let sampler = Footprint.Sampler()
    let pipe = try loadPipeline(rest[0], rest[1], dtypeName == "float32" ? .float32 : .shipped)
    let physAfterLoad = sampler.mark()
    sampler.resetPeak()
    let prompts = (try ConfigIO.json(url(rest[2]))["prompts"] as! [[String: Any]])
    guard let row = prompts.first(where: { ($0["id"] as! String).hasPrefix(rest[3]) }) else { print("no row \(rest[3])"); exit(2) }
    let tags = (row["style"] as! String).split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces).lowercased() }.joined(separator: ",")
    let req = MusicRequest(tags: tags, lyrics: row["lyrics"] as! String, maxSeconds: maxSeconds, seed: seed,
                           sampling: SamplingConfig(cfgScale: cfgScale))
    log("render \(row["id"]!) cap \(maxSeconds)s seed \(seed) cfg \(cfgScale); tags \(tags)")
    let t0 = Date()
    var lastLog = Date()
    let r = try pipe.generate(req) { stage, i, n in
        if Date().timeIntervalSince(lastLog) > 15 { log("  \(stage) \(i)/\(n)"); lastLog = Date() }
        return true
    }
    let wall = Date().timeIntervalSince(t0)
    let peak = sampler.stop()
    let a = r.audio.transposed(1, 0).asArray(Float.self)
    try WAV.writeFloat32(a, channels: 2, sampleRate: 48_000, to: url(rest[4]))
    let finite = a.allSatisfy { $0.isFinite }
    let rms = sqrt(a.reduce(0) { $0 + Double($1) * Double($1) } / Double(a.count))
    let out: [String: Any] = [
        "id": row["id"]!, "seed": seed, "cfg": cfgScale, "cap_s": maxSeconds, "frames": r.frames, "ended_on_end_token": r.endedOnEndToken,
        "effective_duration_s": r.effectiveSeconds, "prompt_tokens": r.promptTokens, "wall_s": wall, "rtf": r.effectiveSeconds / wall,
        "lm_s": r.lmSeconds, "lm_frames_per_s": Double(r.frames) / r.lmSeconds, "prefill_s": r.prefillSeconds, "dit_s": r.ditSeconds,
        "sq_and_assemble_s": r.sqSeconds, "raw_peak": r.rawPeak, "limiter_gain": r.limiterGain, "rms": rms, "finite": finite,
        "phys_after_load_gb": Double(physAfterLoad) / 1e9, "phys_peak_gb": Double(peak) / 1e9,
        "mlx_peak_gb": Double(Memory.peakMemory) / 1e9, "mlx_active_gb": Double(Memory.activeMemory) / 1e9, "dtype": dtypeName,
    ]
    let json = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
    print("RENDER " + String(data: json, encoding: .utf8)!)
    record("render", finite && r.effectiveSeconds > 0 && abs(r.audio).max().item(Float.self) <= Limiter.ceiling + 1e-6,
           "\(r.frames) frames, \(r.endedOnEndToken ? "end token" : "cap"), \(String(format: "%.2f", r.effectiveSeconds)) s, finite \(finite), phys \(String(format: "%.2f", Double(physAfterLoad) / 1e9)) / \(String(format: "%.2f", Double(peak) / 1e9)) GB")
}

do {
    switch cmd {
    case "contract": try contract()
    case "convcheck": try convcheck()
    case "tokenizer": try tokenizerLane()
    case "core": try core()
    case "render": try render()
    default: print("unknown command \(cmd)"); exit(2)
    }
} catch {
    print("ERROR: \(error)"); exit(3)
}
finish()
