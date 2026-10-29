// V3.swift — the engine-package lanes (WIP/heartmula-port/MEASUREMENTS.md §V3, pre-registered 2026-10-28 16:55):
//   validate  <lm> <codec> <prompts.json> <id> <outdir> [--seed N] [--max 120]   MLXServeEngine (default policy) register →
//             prepare → run one song → refusals → evict, phys_footprint at each step; the pairing comes from the bundles
//   envelope  <lm> <codec> <prompts.json>   the 240 s worst case: the longest lyrics, the end token ignored (3 000 frames),
//             then the codec over all of it; phys_footprint peak
//   tf32probe                               T1: what MLX latched (GPU vs CPU fp32 matmul relL2) under this process's environment
import Foundation
import HeartMuLaCore
import MLX
import MLXHeartMuLa
import MLXServeCore
import MLXToolKit

private var v3Failures = 0
private func check(_ name: String, _ ok: Bool, _ detail: String = "") {
    if !ok { v3Failures += 1 }
    print("  \(name): \(ok ? "PASS" : "FAIL") \(detail)")
}
private func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
private func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

private func pcm16(_ audio: Audio) -> [Float] {
    audio.data.dropFirst(44).withUnsafeBytes { raw -> [Float] in raw.bindMemory(to: Int16.self).map { Float($0) / 32768 } }
}

/// The pairing a bundle pair is: the LM's `quantization` block (8-bit → `.int8`, none → `.bf16`) and the codec's dtype.
private func pairing(_ lm: URL, _ codec: URL) throws -> (Quant, HeartMuLaCodecPrecision) {
    let q: Quant = try LMQuantization.from(config: lm.appendingPathComponent("config.json")).map { $0.bits == 8 ? .int8 : .int4 } ?? .bf16
    let c: HeartMuLaCodecPrecision = codec.lastPathComponent.contains("fp16") ? .fp16 : .fp32
    return (q, c)
}

private func row(_ prompts: URL, _ id: String) throws -> [String: Any] {
    let rows = try ConfigIO.json(prompts)["prompts"] as! [[String: Any]]
    guard let r = rows.first(where: { ($0["id"] as! String).hasPrefix(id) }) else { throw HeartMuLaError.invalidInput("no row \(id)") }
    return r
}

func runValidate(_ lm: URL, _ codec: URL, _ prompts: URL, _ id: String, _ outDir: URL, seed: UInt64, cap: Double) async throws -> Int32 {
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let r = try row(prompts, id)
    let (quant, codecPrecision) = try pairing(lm, codec)
    let tier = "\(quant)-\(codecPrecision.rawValue)"
    print("validate: MLXServeEngine → register → prepare → run (LM \(quant), codec \(codecPrecision.rawValue); MLX_ENABLE_TF32=\(ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset")) → refusals → evict")
    let sampler = Footprint.Sampler()
    let before = sampler.mark()
    let engine = MLXServeEngine()   // the default policy (.permissiveOnly, .advisory): Apache-2.0 registers with no advisory
    let t0 = Date()
    let configuration = HeartMuLaConfiguration(quant: quant, codec: codecPrecision, lmDirectory: lm, codecDirectory: codec,
                                               cacheLimitBytes: 1 << 30)
    let pid = try await engine.register(HeartMuLaPackage.registration, configuration: configuration, id: PackageID("heartmula-oss-3b"))
    let advisories = await engine.licenseAdvisories
    check("registered \(pid.rawValue); licence advisories: \(advisories.count)", advisories.isEmpty, advisories.map(\.summary).joined(separator: "; "))
    let needs = await engine.needsDownload(.music, package: pid)
    check("needsDownload == \(needs) with the explicit bundles", !needs)
    try await engine.prepare(.music, package: pid)
    Thread.sleep(forTimeInterval: 0.3)
    let afterLoad = sampler.mark()
    let charged = await engine.memory.residentBytes
    print(String(format: "  [VAL] prepared in %.1fs; phys_footprint %@ → %@ (resident +%@); MLX active %@; engine charges %@; TF32 probe relL2 %.1e",
                 Date().timeIntervalSince(t0), gb(before) as NSString, gb(afterLoad) as NSString, gb(Int(afterLoad) - Int(before)) as NSString,
                 gb(Memory.activeMemory) as NSString, gb(charged) as NSString, TF32.probeRelL2()))
    sampler.resetPeak()
    let caption = r["style"] as! String, lyrics = r["lyrics"] as! String
    let request = MusicRequest(caption: caption, lyrics: lyrics, durationSeconds: cap, seed: seed)
    var phases: [String: Int] = [:]
    var maxStage = 0
    let t = Date()
    let response = try await RunProgress.$sink.withValue({ report in
        phases[report.phase.rawValue, default: 0] += 1
        if report.phase == .denoise { maxStage = max(maxStage, (report.stage ?? 0) + 1) }
    }) {
        try await engine.run(request, package: pid)
    }
    let wall = Date().timeIntervalSince(t)
    Thread.sleep(forTimeInterval: 0.3)
    let peak = max(afterLoad, sampler.peak)
    let afterSong = sampler.mark()
    guard let m = response as? MusicResponse else { check("MusicResponse", false, "got \(type(of: response))"); return 1 }
    let samples = pcm16(m.audio)
    let secs = Double(samples.count) / Double((m.audio.sampleRate ?? 48_000) * (m.audio.channels ?? 2))
    let finite = samples.allSatisfy(\.isFinite)
    let rms = 20 * log10(max(sqrt(samples.map { $0 * $0 }.reduce(0, +) / Float(max(samples.count, 1))), 1e-9))
    let peakLin = samples.map(abs).max() ?? 0
    let peakDB = 20 * log10(max(peakLin, 1e-9))
    let consistent = m.truncated ? secs >= cap - 1 : secs < cap
    let windows = max(maxStage, 1)
    check(String(format: "[RUN] %.2fs audio of a %.0f s cap (%d Hz × %d ch), peak %.2f dBFS, rms %.1f dBFS, %.1fs wall (RTF %.2f), truncated %@, phases %@, %d codec windows, phys peak %@ (after %@)",
                 secs, cap, m.audio.sampleRate ?? 0, m.audio.channels ?? 0, peakDB, rms, wall, secs / max(wall, 1e-9), "\(m.truncated)" as NSString,
                 "\(phases.sorted { $0.key < $1.key })" as NSString, windows, gb(peak) as NSString, gb(afterSong) as NSString),
          finite && secs > 10 && secs <= cap && peakLin <= Limiter.ceiling + 1 / 32768 && rms > -40 && rms < -3
              && m.audio.sampleRate == 48_000 && m.audio.channels == 2 && consistent
              && (phases["generate"] ?? 0) > 100 && (phases["denoise"] ?? 0) >= 10 * windows && (phases["decode"] ?? 0) >= 1
              && (phases["postprocess"] ?? 0) == 1)
    try m.audio.data.write(to: outDir.appendingPathComponent("\(id)__engine_\(tier).wav"))
    // undeclared features are refused BEFORE admission (MusicControls.refusal); the package refuses its own bounds in run()
    let clip = Audio(format: .wav, data: Data([1, 2, 3]))
    for (label, bad) in [
        ("planning full (undeclared)", MusicRequest(caption: caption, lyrics: lyrics, planning: .full)),
        ("score (undeclared)", MusicRequest(caption: caption, lyrics: lyrics, score: "X:1\nK:C\nCDEF|")),
        ("edit/cover (undeclared)", MusicRequest(caption: caption, lyrics: lyrics, edit: MusicEdit(source: clip, operation: .cover(strength: 0.5)))),
        ("referenceAudio (undeclared)", MusicRequest(caption: caption, lyrics: lyrics, referenceAudio: clip)),
        ("vocalLanguage en (undeclared)", MusicRequest(caption: caption, lyrics: lyrics, vocalLanguage: "en")),
        ("anchors bpm 88 (undeclared)", MusicRequest(caption: caption, lyrics: lyrics, anchors: MusicAnchors(bpm: 88))),
        ("durationSeconds 400 (> 240)", MusicRequest(caption: caption, lyrics: lyrics, durationSeconds: 400)),
        ("durationSeconds 0.5 (< 1)", MusicRequest(caption: caption, lyrics: lyrics, durationSeconds: 0.5)),
        ("empty lyrics", MusicRequest(caption: caption, lyrics: " ")),
        ("empty caption", MusicRequest(caption: " , ", lyrics: lyrics)),
    ] {
        do { _ = try await engine.run(bad, package: pid); check("refused: \(label)", false, "ADMITTED") }
        catch { check("refused: \(label)", error is PackageError, String(describing: error).prefix(120).description) }
    }
    await engine.evict(package: pid)
    Thread.sleep(forTimeInterval: 0.5)
    let afterEvict = sampler.mark()
    _ = sampler.stop()
    check("after evict \(gb(afterEvict)) < 1.5 GB", afterEvict < 1_500_000_000)
    print(String(format: "  [VAL] %@: resident %@ (phys after prepare); after the song %@; peak %@ (activation %@ over the post-prepare resident); after evict %@; MLX peak %@",
                 tier as NSString, gb(afterLoad) as NSString, gb(afterSong) as NSString, gb(peak) as NSString,
                 gb(Int(peak) - Int(afterLoad)) as NSString, gb(afterEvict) as NSString, gb(Memory.peakMemory) as NSString))
    let out: [String: Any] = ["tier": tier, "phys_before": before, "phys_after_prepare": afterLoad, "engine_charged": charged,
                              "phys_after_song": afterSong, "phys_peak": peak, "phys_after_evict": afterEvict, "mlx_peak": Memory.peakMemory,
                              "audio_s": secs, "truncated": m.truncated, "wall_s": wall, "rms_dbfs": rms, "peak_dbfs": peakDB,
                              "phases": phases, "codec_windows": windows, "failures": v3Failures]
    print("VALIDATE " + String(data: try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]), encoding: .utf8)!)
    print(v3Failures == 0 ? "RESULT: PASS" : "RESULT: FAIL (\(v3Failures) checks)")
    return v3Failures == 0 ? 0 : 1
}

func runEnvelope(_ lm: URL, _ codec: URL, _ prompts: URL) throws {
    let (quant, codecPrecision) = try pairing(lm, codec)
    let rows = try ConfigIO.json(prompts)["prompts"] as! [[String: Any]]
    let longest = rows.max { ($0["lyrics"] as! String).count < ($1["lyrics"] as! String).count }!
    let sampler = Footprint.Sampler()
    let before = sampler.mark()
    TF32.disableForProcess()
    Memory.cacheLimit = 1 << 30
    let t0 = Date()
    let pipe = try HeartMuLaPipeline(lmDir: lm, codecDir: codec, dtype: .shipped)   // exactly as HeartMuLaPackage.load
    Memory.clearCache()
    Thread.sleep(forTimeInterval: 0.3)
    let afterLoad = sampler.mark()
    print(String(format: "envelope (LM %@, codec %@): loaded in %.1fs; phys %@ → %@", "\(quant)" as NSString, codecPrecision.rawValue as NSString,
                 Date().timeIntervalSince(t0), gb(before) as NSString, gb(afterLoad) as NSString))
    sampler.resetPeak()
    let req = SongRequest(tags: HeartMuLaPackage.tags(from: longest["style"] as! String), lyrics: longest["lyrics"] as! String,
                          maxSeconds: SongRequest.lmSeconds(cappingAudioAt: HeartMuLaPackage.maxDurationSeconds), seed: 1,
                          stopOnEndToken: false)
    let t1 = Date()
    let r = try pipe.generate(req)
    Thread.sleep(forTimeInterval: 0.3)
    let peak = max(afterLoad, sampler.stop())
    let out: [String: Any] = ["tier": "\(quant)-\(codecPrecision.rawValue)", "id": longest["id"]!, "prompt_tokens": r.promptTokens,
                              "frames": r.frames, "audio_s": r.effectiveSeconds, "wall_s": Date().timeIntervalSince(t1),
                              "codec_windows": r.ditSeconds.count, "phys_after_load": afterLoad, "phys_peak": peak,
                              "activation": Int(peak) - Int(afterLoad), "mlx_peak": Memory.peakMemory]
    print(String(format: "  %@: prompt %d + %d frames → %.2f s of audio (%d codec windows) in %.1f s; phys peak %@ (activation %@ over the resident); MLX peak %@",
                 longest["id"] as! String as NSString, r.promptTokens, r.frames, r.effectiveSeconds, r.ditSeconds.count, Date().timeIntervalSince(t1),
                 gb(peak) as NSString, gb(Int(peak) - Int(afterLoad)) as NSString, gb(Memory.peakMemory) as NSString))
    print("ENVELOPE " + String(data: try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys]), encoding: .utf8)!)
}

func runTF32Probe() {
    let env = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset"
    let rel = TF32.probeRelL2()
    print(String(format: "TF32PROBE MLX_ENABLE_TF32=%@ relL2 %.3e → %@ (threshold %.0e)", env as NSString, rel,
                 rel < TF32.offThreshold ? "off" : "on", TF32.offThreshold))
}

/// The latch policy, live: a matmul FIRST (latching whatever MLX_ENABLE_TF32 says), then MLXServeEngine register → prepare. With
/// TF32 latched on, the fp32 codec must be refused with the fix named, and the fp16 codec must load (V3 T2: TF32-immune).
func runTF32Policy(_ lm: URL, _ codec: URL) async -> Int32 {
    let (quant, codecPrecision) = (try? pairing(lm, codec)) ?? (.int8, .fp16)
    let rel = TF32.probeRelL2()   // latches the process's TF32 setting before the package can set it
    print(String(format: "tf32policy: MLX_ENABLE_TF32=%@ latched → probe %.1e (%@); LM %@, codec %@",
                 (ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] ?? "unset") as NSString, rel, rel < TF32.offThreshold ? "off" : "on",
                 "\(quant)" as NSString, codecPrecision.rawValue as NSString))
    let engine = MLXServeEngine()
    do {
        let pid = try await engine.register(HeartMuLaPackage.registration,
                                            configuration: HeartMuLaConfiguration(quant: quant, codec: codecPrecision, lmDirectory: lm, codecDirectory: codec),
                                            id: PackageID("heartmula-oss-3b"))
        try await engine.prepare(.music, package: pid)
        print("TF32POLICY prepared (loaded)")
        await engine.evict(package: pid)
    } catch {
        print("TF32POLICY refused: \(error)")
    }
    return 0
}
