// StructuralTests.swift — host-side invariants that need no weights and no Metal: the codec windowing arithmetic, the
// code-repeat rule, and the torchtune θ table's shape. The weight-bearing gates are `heartmula-gates` lanes.
import Foundation
import HeartMuLaCore
import Testing

@Test func windowingMatchesUpstream() {
    #expect(CodecWindowing.minFrames == 372)
    #expect(CodecWindowing.hopFrames == 320)
    #expect(CodecWindowing.overlapLatents == 104)
    #expect(CodecWindowing.latentLength == 744)
    #expect(CodecWindowing.windowSamples == 1_428_480)
    #expect(CodecWindowing.hopSamples == 1_228_800)
    #expect(CodecWindowing.overlapSamples == 199_680)
    // 401 frames (the goldens): padded by doubling to 692 = 2 windows; 1 539 840 samples
    let idx = CodecWindowing.paddedIndices(frames: 401)
    #expect(idx.count == 692)
    #expect(Array(idx[401 ..< 410]) == Array(0 ..< 9))
    #expect(CodecWindowing.windowStarts(paddedLength: idx.count) == [0, 320])
    #expect(CodecWindowing.targetSamples(frames: 401) == 1_539_840)
    // shorter than one window: repeated to 372, one window
    let short = CodecWindowing.paddedIndices(frames: 100)
    #expect(short.count == 372 && short[100] == 0 && short[371] == 71)
    #expect(CodecWindowing.windowStarts(paddedLength: short.count) == [0])
}

@Test func ropeThetaShape() {
    let b = LlamaFlavor.llama3B.ropeTheta(), d = LlamaFlavor.llama300M.ropeTheta()
    #expect(b.count == 64 && d.count == 192)
    #expect(b[0] == 1)
    #expect(zip(b, b.dropFirst()).allSatisfy { $0 >= $1 })
}
