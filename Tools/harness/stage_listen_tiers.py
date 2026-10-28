#!/usr/bin/env python3
"""E27 V2b: stage draw-A takes of every tier beside the V0 quadruples for the operator's ear (evidence, not a gate):
symlinks WIP/acestep-eval/_out/listen/<id>__heartmula-swift-<tier>.wav -> _out/heartmula-swift-<tier>-a/<id>.wav."""
import pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C
LISTEN = C.ACE_OUT / "listen"
n = 0
for tier in ("bf16", "lm8", "lm4", "c16", "cfg3"):
    for p in C.prompts():
        src = C.OUT / f"heartmula-swift-{tier}-a" / f"{p['id']}.wav"
        if src.exists():
            dst = LISTEN / f"{p['id']}__heartmula-swift-{tier}.wav"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src); n += 1
print(f"staged {n} takes in {LISTEN}")
