// ConformanceTests.swift — the offline conformance gate for MLXHeartMuLa (WIP/heartmula-port/MEASUREMENTS.md §V3): manifest (C0 /
// C1 / C7 / C8 / C-memory / C11 / provenance; Apache-2.0 admitted with no advisory), the MAT gate per offered pairing (two roles),
// the CAN gate (pre-cancelled run propagation, checkpoint cadence), the unloaded and unoffered-tier refusals, and the request
// plane (the core fields, the caption → tags mapping, the honest cap, every undeclared feature refused). Nothing here evals a
// kernel: the live run is `heartmula-gates validate` (GPU, through MLXServeEngine).

import Foundation
import MLXServeConformance
import MLXServeCore
import MLXToolKit
import XCTest

@testable import HeartMuLaCore
@testable import MLXHeartMuLa

final class ManifestConformanceTests: XCTestCase {

    /// C7/C8 — Apache-2.0 weights, MIT port code; admitted by `.permissiveOnly`; no required display notice; the grant chain and
    /// the measured character as informational notices.
    func testLicenseIsApacheWithTheGrantChainAndCharacter() {
        let license = HeartMuLaPackage.manifest.license
        XCTAssertEqual(license.weightLicense, .apache2)
        XCTAssertNil(license.additionalWeightLicenses)
        XCTAssertEqual(license.portCodeLicense, .mit)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(license).isAdmitted)
        XCTAssertTrue(SPDXLicense.permissiveAllowlist.contains(license.weightLicense))
        XCTAssertTrue(license.missingRequiredDisplayNotices.isEmpty)
        XCTAssertTrue(license.requiredNotices.isEmpty)
        let notices = license.notices ?? []
        XCTAssertEqual(notices.count, 3)
        XCTAssertTrue(notices.allSatisfy { $0.display == .informational })
        let text = notices.map(\.text).joined(separator: " ")
        XCTAssertTrue(text.contains("no LICENSE file"))
        XCTAssertTrue(text.contains("2026-01-20"))
        XCTAssertTrue(text.contains("Llama-3.2"))
        // the character the operator asked to declare (AB-D-0187, V2b)
        XCTAssertTrue(text.contains("style tags"))
        XCTAssertTrue(text.contains("about 4.5"))
        XCTAssertTrue(text.contains("male-voice") || text.contains("female"))
        XCTAssertTrue(text.contains("Mandarin"))
        XCTAssertTrue(text.contains("10 of 24"))
        XCTAssertTrue(text.contains("looped"))
        XCTAssertFalse(HeartMuLaPackage.namedOnly)
        XCTAssertFalse(HeartMuLaPackage.acceptedRequired)
        XCTAssertTrue(HeartMuLaPackage.commercialUse)
    }

    /// The engine's default policy registers it with no advisory, and so does `.blocking`.
    func testEngineGateAdmitsTheWeights() async throws {
        for engine in [MLXServeEngine(), MLXServeEngine(licenseEnforcement: .blocking)] {
            _ = try await engine.register(HeartMuLaPackage.registration, configuration: HeartMuLaConfiguration(),
                                          id: PackageID("heartmula-oss-3b"))
            let advisories = await engine.licenseAdvisories
            XCTAssertTrue(advisories.isEmpty, advisories.map(\.summary).joined(separator: "; "))
        }
    }

    /// C-memory — split footprints for the two LM tiers (fp16-codec pairings), at least what `validate` / `envelope` / V2b
    /// measured; the fp32-codec pairings hint their own bytes.
    func testFootprintsAreTheMeasuredPairings() {
        let footprints = HeartMuLaPackage.manifest.requirements.footprints
        XCTAssertEqual(footprints.map(\.quant), [.bf16, .int8])
        for (quant, codec, resident, peak) in Measured.pairings {
            let c = HeartMuLaConfiguration(quant: quant, codec: codec)
            let fp = footprints.first { $0.quant == quant }!
            let r = c.residentBytesHint ?? fp.residentBytes
            let a = c.peakActivationBytesHint ?? fp.peakActivationBytes
            XCTAssertGreaterThanOrEqual(r, resident, "\(quant)-\(codec) resident")
            XCTAssertGreaterThanOrEqual(r + a, peak, "\(quant)-\(codec) peak")
        }
        XCTAssertNil(HeartMuLaConfiguration().residentBytesHint)
        XCTAssertNil(HeartMuLaConfiguration(quant: .bf16).peakActivationBytesHint)
        XCTAssertNotNil(HeartMuLaConfiguration(codec: .fp32).residentBytesHint)
        XCTAssertNotNil(HeartMuLaConfiguration(quant: .bf16, codec: .fp32).peakActivationBytesHint)
        XCTAssertLessThan(footprints[1].residentBytes, footprints[0].residentBytes)
        XCTAssertTrue(CancellationConformance.longRunImplied(by: HeartMuLaPackage.manifest))
        XCTAssertEqual(HeartMuLaPackage.offeredQuants, [.int8, .bf16])
        XCTAssertEqual(HeartMuLaPackage.offeredCodecs, [.fp16, .fp32])
        XCTAssertEqual(HeartMuLaConfiguration().quant, .int8)
        XCTAssertEqual(HeartMuLaConfiguration().codec, .fp16)
        XCTAssertFalse(HeartMuLaPackage.tf32ImmuneCodecs.contains(.fp32))
    }

    /// C1 / C11 — `music` only; the surface declares the core and nothing else; a 240 s cap.
    func testCapabilityIsMusicWithAnHonestDeclaration() {
        XCTAssertEqual(Set(HeartMuLaPackage.manifest.capabilities), [.music])
        let surface = HeartMuLaPackage.manifest.surfaces.first { $0.capability == .music }
        XCTAssertEqual(surface?.name, "heartmula-oss-3b")
        let controls = surface?.musicControls
        XCTAssertEqual(controls?.features, [])
        XCTAssertEqual(controls?.planningModes, [])
        XCTAssertEqual(controls?.editModes, [])
        XCTAssertEqual(controls?.vocalLanguages, [])
        XCTAssertEqual(controls?.maxDurationSeconds, 240)
        XCTAssertEqual(controls?.declarationProblems, [])
        XCTAssertEqual(Set(surface?.parameters.map(\.name) ?? []), ["caption", "lyrics", "durationSeconds", "seed"])
        XCTAssertTrue(surface?.summary.contains("cap") ?? false)
        XCTAssertTrue(surface?.summary.contains("Apache-2.0") ?? false)
    }

    /// C0 — the manifest targets a contract that has the music capability (1.56.0 or later).
    func testContractVersionHasTheMusicCapability() {
        let v = HeartMuLaPackage.manifest.contractVersion
        XCTAssertTrue(v.major > 1 || (v.major == 1 && v.minor >= 56), "\(v)")
    }

    func testProvenancePinsTheUpstreamCheckpoint() {
        let p = HeartMuLaPackage.manifest.provenance
        XCTAssertEqual(p.sourceRepo, "HeartMuLa/HeartMuLa-oss-3B-happy-new-year")
        XCTAssertEqual(p.revision, "41f6fc68490e11dc43fdabaa6b5767946408c903")
        XCTAssertEqual(p.tier, 3)
        XCTAssertEqual(HeartMuLaConfiguration.upstreamCodecRevision.count, 40)
        XCTAssertEqual(HeartMuLaConfiguration.upstreamGenRevision.count, 40)
    }
}

final class MaterializationConformanceTests: XCTestCase {

    private static let pairings: [(Quant, HeartMuLaCodecPrecision)] = [(.int8, .fp16), (.int8, .fp32), (.bf16, .fp16), (.bf16, .fp32)]

    /// MAT-1..5: a fresh (dir-less) configuration reports both sources missing; explicit bundles satisfy — per offered pairing.
    func testMaterializationGatePerPairing() throws {
        for (q, c) in Self.pairings {
            let (satisfied, root) = try satisfiedConfiguration(quant: q, codec: c)
            defer { try? FileManager.default.removeItem(at: root) }
            let report = MaterializationConformance.check(
                freshConfiguration: HeartMuLaConfiguration(quant: q, codec: c, modelsRootDirectory: emptyStoreRoot()),
                satisfiedConfiguration: satisfied)
            XCTAssertTrue(report.passed, "\(q)-\(c): \(report.summary)")
        }
    }

    /// Two roles: the LM bundle (bf16 or the pre-quantised 8-bit) and the codec bundle (fp32 or fp16) — the V4 candidate repos.
    func testWeightSources() {
        let expect: [(Quant, HeartMuLaCodecPrecision, [String])] = [
            (.int8, .fp16, ["mlx-community/HeartMuLa-oss-3B-8bit", "mlx-community/HeartCodec-oss-fp16"]),
            (.int8, .fp32, ["mlx-community/HeartMuLa-oss-3B-8bit", "mlx-community/HeartCodec-oss-fp32"]),
            (.bf16, .fp16, ["mlx-community/HeartMuLa-oss-3B-bf16", "mlx-community/HeartCodec-oss-fp16"]),
            (.bf16, .fp32, ["mlx-community/HeartMuLa-oss-3B-bf16", "mlx-community/HeartCodec-oss-fp32"]),
        ]
        for (q, c, repos) in expect {
            let cfg = HeartMuLaConfiguration(quant: q, codec: c)
            XCTAssertEqual(cfg.weightSources.map(\.role), ["lm", "codec"])
            XCTAssertEqual(cfg.weightSources.map(\.repo), repos)
            XCTAssertEqual(cfg.weightSources[0].matching, HeartMuLaConfiguration.lmFiles)
            XCTAssertEqual(cfg.weightSources[1].matching, HeartMuLaConfiguration.codecFiles)
        }
        XCTAssertNil(HeartMuLaConfiguration(quant: .int4).lmRepo)
        XCTAssertEqual(HeartMuLaConfiguration(revision: "abc", codecRevision: "def").weightSources.map(\.revision), ["abc", "def"])
        XCTAssertEqual(HeartMuLaConfiguration().cacheLimitBytes, 1 << 30)
    }

    /// The declared globs are the files `HeartMuLaPipeline` opens, in each bundle layout the converter writes.
    func testDeclaredFilesMatchTheBundleLayouts() {
        let lmSharded = ["LICENSE", "NOTICE", "config.json", "gen_config.json", "model-00001-of-00002.safetensors",
                         "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json"]
        let lmSingle = ["LICENSE", "NOTICE", "config.json", "gen_config.json", "model.safetensors", "tokenizer.json"]
        let codecSharded = ["LICENSE", "NOTICE", "config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                            "model.safetensors.index.json"]
        let codecSingle = ["LICENSE", "NOTICE", "config.json", "model.safetensors"]
        for layout in [lmSharded, lmSingle] {
            for glob in HeartMuLaConfiguration.lmFiles {
                XCTAssertTrue(layout.contains { WeightSourceProbe.matches(path: $0, glob: glob) }, "lm glob \(glob)")
            }
        }
        for layout in [codecSharded, codecSingle] {
            for glob in HeartMuLaConfiguration.codecFiles {
                XCTAssertTrue(layout.contains { WeightSourceProbe.matches(path: $0, glob: glob) }, "codec glob \(glob)")
            }
        }
    }

    /// An explicit pair reads as missing per role until its probe file exists; resolution needs both.
    func testExplicitBundleProbe() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("heartmula-partial-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let l = root.appendingPathComponent("lm"), c = root.appendingPathComponent("codec")
        try FileManager.default.createDirectory(at: l, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: c, withIntermediateDirectories: true)
        let cfg = HeartMuLaConfiguration(lmDirectory: l, codecDirectory: c)
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: nil).map(\.role), ["lm", "codec"])
        try Data().write(to: l.appendingPathComponent(HeartMuLaConfiguration.lmProbe))
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: nil).map(\.role), ["codec"])
        try Data().write(to: c.appendingPathComponent(HeartMuLaConfiguration.codecProbe))
        XCTAssertTrue(cfg.missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(cfg.resolvedBundles(storeRoot: nil)?.lm, l)
        XCTAssertEqual(cfg.resolvedBundles(storeRoot: nil)?.codec, c)
        XCTAssertEqual(cfg.prewarmPaths.count, 2)
        XCTAssertNil(HeartMuLaConfiguration().resolvedBundles(storeRoot: nil))
        XCTAssertEqual(HeartMuLaConfiguration(lmDirectory: l).missingWeightSources(storeRoot: nil).map(\.role), ["codec"])
    }

    func testCodableExcludesEnvironmentURLs() throws {
        let c = HeartMuLaConfiguration(quant: .bf16, codec: .fp32, revision: "abc", codecRevision: "def",
                                       lmDirectory: URL(fileURLWithPath: "/tmp/l"), codecDirectory: URL(fileURLWithPath: "/tmp/c"),
                                       modelsRootDirectory: URL(fileURLWithPath: "/tmp/y"), cacheLimitBytes: 2 << 30)
        let decoded = try JSONDecoder().decode(HeartMuLaConfiguration.self, from: JSONEncoder().encode(c))
        XCTAssertNil(decoded.lmDirectory)
        XCTAssertNil(decoded.codecDirectory)
        XCTAssertNil(decoded.modelsRootDirectory)
        XCTAssertEqual(decoded.quant, .bf16)
        XCTAssertEqual(decoded.codec, .fp32)
        XCTAssertEqual(decoded.revision, "abc")
        XCTAssertEqual(decoded.codecRevision, "def")
        XCTAssertEqual(decoded.cacheLimitBytes, 2 << 30)
        let old = try JSONDecoder().decode(HeartMuLaConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(old.quant, .int8)
        XCTAssertEqual(old.codec, .fp16)
    }

    private func emptyStoreRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("heartmula-empty-store-\(UUID().uuidString)")
    }

    private func satisfiedConfiguration(quant: Quant, codec: HeartMuLaCodecPrecision) throws -> (HeartMuLaConfiguration, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("heartmula-explicit-\(UUID().uuidString)")
        let l = root.appendingPathComponent("lm"), c = root.appendingPathComponent("codec")
        try FileManager.default.createDirectory(at: l, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: c, withIntermediateDirectories: true)
        try Data().write(to: l.appendingPathComponent(HeartMuLaConfiguration.lmProbe))
        try Data().write(to: c.appendingPathComponent(HeartMuLaConfiguration.codecProbe))
        return (HeartMuLaConfiguration(quant: quant, codec: codec, lmDirectory: l, codecDirectory: c), root)
    }
}

final class CancellationConformanceTests: XCTestCase {

    private func probe() -> MusicRequest {
        MusicRequest(caption: "cancellation probe", lyrics: "[verse]\nla la la", durationSeconds: 30, seed: 1)
    }

    /// CAN-1/CAN-2 — a pre-cancelled `run()` surfaces `CancellationError` unchanged, even unloaded.
    func testPreCancelledRunPropagatesCancellation() async {
        let package = HeartMuLaPackage(configuration: HeartMuLaConfiguration())
        let report = await CancellationConformance.checkRun(package: package, request: probe())
        XCTAssertTrue(report.passed, report.summary)
    }

    /// CAN-3 — checkpoints at every LM frame (generate), every Euler step (denoise) and every SQ-Codec window (decode), each on the
    /// same seam as the RunProgress report.
    func testCheckpointCadence() {
        let report = CancellationConformance.checkCadence(
            manifest: HeartMuLaPackage.manifest,
            posture: .cadence([.init(phase: .generate, unit: .token, reportsRunProgress: true),
                               .init(phase: .denoise, unit: .step, reportsRunProgress: true),
                               .init(phase: .decode, unit: .chunk, reportsRunProgress: true)]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

final class RequestPlaneTests: XCTestCase {

    private func request(duration: Double? = 120, caption: String = "Folk, Acoustic Guitar , female vocal", lyrics: String = "[verse]\nla la la",
                         language: String? = nil, anchors: MusicAnchors? = nil, planning: MusicPlanningMode? = nil, score: String? = nil,
                         edit: MusicEdit? = nil, reference: Audio? = nil, guidance: Double? = nil, seed: UInt64? = 7) -> MusicRequest {
        MusicRequest(caption: caption, lyrics: lyrics, durationSeconds: duration, seed: seed, guidanceScale: guidance,
                     planning: planning, score: score, referenceAudio: reference, edit: edit, anchors: anchors, vocalLanguage: language)
    }

    func testUnloadedPackageRejectsLegibly() async {
        let package = HeartMuLaPackage(configuration: HeartMuLaConfiguration())
        do {
            _ = try await package.run(request())
            XCTFail("expected notLoaded")
        } catch let error as PackageError {
            guard case .notLoaded = error else { return XCTFail("expected notLoaded, got \(error)") }
        } catch {
            XCTFail("expected PackageError, got \(error)")
        }
    }

    /// The 4-bit LM (V2b Q4 FAIL) and every other unoffered quant are refused before any weight is touched.
    func testUnofferedTierIsRefusedByLoad() async {
        for q in [Quant.int4, .fp32] {
            let package = HeartMuLaPackage(configuration: HeartMuLaConfiguration(quant: q))
            do {
                try await package.load()
                XCTFail("loaded \(q)")
            } catch let error as HeartMuLaPackageError {
                guard case .unsupportedTier(let why) = error else { return XCTFail("\(error)") }
                XCTAssertTrue(why.contains("4-bit"))
            } catch {
                XCTFail("\(error)")
            }
        }
    }

    /// The core maps onto the Core's request: the caption becomes the V0 harness's tags, the cap bounds the audio (floor(cap ×
    /// 12.5) frames), nil → 120 s, a nil seed is drawn.
    func testCoreMapsOntoThePipeline() throws {
        let p = try HeartMuLaPackage.plan(request())
        XCTAssertEqual(p.request.tags, "folk,acoustic guitar,female vocal")
        XCTAssertEqual(p.request.lyrics, "[verse]\nla la la")
        XCTAssertEqual(p.request.seed, 7)
        XCTAssertEqual(p.capSeconds, 120)
        XCTAssertEqual(p.request.maxFrames, 1500)                                  // 120.00 s, not heartlib's 1 501 (120.08 s)
        XCTAssertEqual(try HeartMuLaPackage.plan(request(duration: nil)).request.maxFrames, 1500)
        XCTAssertEqual(try HeartMuLaPackage.plan(request(duration: 240)).request.maxFrames, 3000)
        XCTAssertEqual(try HeartMuLaPackage.plan(request(duration: 84.3)).request.maxFrames, 1053)   // floor(1053.75)
        XCTAssertEqual(try HeartMuLaPackage.plan(request(duration: 1)).request.maxFrames, 12)
        for cap in stride(from: 1.0, through: 240, by: 7.3) {
            let f = try HeartMuLaPackage.plan(request(duration: cap)).request.maxFrames
            XCTAssertLessThanOrEqual(Double(f) / 12.5, cap + 1e-9, "cap \(cap)")   // float noise: 190.79999999999998 × 12.5 == 2385
            XCTAssertGreaterThan(Double(f + 1) / 12.5, cap, "cap \(cap)")
        }
        XCTAssertEqual(p.request.sampling.cfgScale, 1.5)
        XCTAssertEqual(p.request.codecSteps, 10)
        XCTAssertEqual(p.request.codecGuidance, 1.25)
        XCTAssertTrue(p.request.stopOnEndToken)
        XCTAssertNotEqual(try HeartMuLaPackage.plan(request(seed: nil)).request.seed, 0)
        XCTAssertNoThrow(try HeartMuLaPackage.plan(request(anchors: MusicAnchors())))   // an empty anchor asks for nothing
        // an undeclared guidanceScale is ignored, not refused (the contract's MusicFeature.guidanceScale rule)
        XCTAssertEqual(try HeartMuLaPackage.plan(request(guidance: 3)).request.sampling.cfgScale, 1.5)
        XCTAssertEqual(HeartMuLaPackage.tags(from: " Jazz ,, Smoky Lounge,"), "jazz,smoky lounge")
    }

    /// Every undeclared feature and out-of-range value is refused legibly (the engine refuses the undeclared ones before admission).
    func testUndeclaredFeaturesAndBoundsAreRefused() {
        let clip = Audio(format: .wav, data: Data([1]))
        let bad: [(String, MusicRequest)] = [
            ("planning full", request(planning: .full)),
            ("planning off", request(planning: .off)),
            ("score", request(score: "X:1\nK:C\nCDEF|")),
            ("edit", request(edit: MusicEdit(source: clip, operation: .cover(strength: 0.5)))),
            ("reference", request(reference: clip)),
            ("anchors", request(anchors: MusicAnchors(bpm: 112))),
            ("language", request(language: "en")),
            ("too long", request(duration: 400)),
            ("over the cap", request(duration: 240.5)),
            ("too short", request(duration: 0.5)),
            ("zero length", request(duration: 0)),
            ("nan length", request(duration: .nan)),
            ("empty caption", request(caption: " , ,")),
            ("empty lyrics", request(lyrics: "  \n")),
        ]
        for (label, r) in bad {
            do { _ = try HeartMuLaPackage.plan(r); XCTFail("admitted: \(label)") }
            catch let e as PackageError { guard case .unsupportedRequestFeature = e else { return XCTFail("\(label): \(e)") } }
            catch { XCTFail("\(label): \(error)") }
        }
        for r in [request(planning: .full), request(language: "en"), request(duration: 400), request(anchors: MusicAnchors(key: "C major")),
                  request(reference: clip), request(score: "X:1")] {
            XCTAssertNotNil(HeartMuLaPackage.controls.refusal(for: r))
        }
        XCTAssertNil(HeartMuLaPackage.controls.refusal(for: request(guidance: 3)))
    }

    func testWAVEncoding() {
        let data = HeartMuLaAudioIO.encodeWAV16(interleaved: [0, 0.5, -0.5, 2, -2, .nan], channels: 2, sampleRate: 48_000)
        XCTAssertEqual(data.count, 44 + 12)
        XCTAssertEqual(String(decoding: data[0 ..< 4], as: UTF8.self), "RIFF")
        XCTAssertEqual(data[22], 2)
        XCTAssertEqual(data[24 ..< 28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }, 48_000)
        let pcm = data.dropFirst(44).withUnsafeBytes { raw in raw.bindMemory(to: Int16.self).map { $0 } }
        XCTAssertEqual(pcm, [0, 16383, -16383, 32767, -32767, 0])
    }
}

/// The phys_footprint readings the declarations must cover (MEASUREMENTS §V3): (LM, codec, resident after prepare, peak).
enum Measured {
    static let pairings: [(Quant, HeartMuLaCodecPrecision, UInt64, UInt64)] = [
        (.int8, .fp16, 8_320_000_000, 13_770_000_000),     // validate resident; V2b lm8 + fp16 12-song peak (V3 song 12.82, envelope 12.80)
        (.int8, .fp32, 11_470_000_000, 21_030_000_000),    // validate re-time resident; V2b lm8 12-song peak (V3 19.60 / 18.40)
        (.bf16, .fp16, 11_260_000_000, 16_790_000_000),    // validate resident; V2b c16 decode process peak (V3 15.78 / 15.68)
        (.bf16, .fp32, 14_400_000_000, 24_030_000_000),    // validate resident; V2b bf16 12-song peak (V3 21.22 / 21.61)
    ]
}
