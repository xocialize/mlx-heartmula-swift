#!/usr/bin/env python3
"""E27 V0 style-fit (reported, not a gate condition; rules pre-registered in MEASUREMENTS.md §Gate before any number existed).
Same computation for every system, from the audio already on disk (no re-render):
  SF-CLAP   laion/larger_clap_music_and_speech (transformers 4.57 ClapModel; the pre-registered larger_clap_music is degenerate): song embedding = mean of L2-normalised 10 s window embeddings
            (48 kHz mono, windows < 5 s dropped); text = the row's `style`; 12-way retrieval -> top-1, rank of the true style,
            margin (own cosine - mean of the other 11).
  SF-gender median librosa.pyin F0 (65-1000 Hz) over voiced frames of the scorer's Demucs vocals stem; >= 165 Hz = female.
  SF-tempo  librosa.beat.beat_track on the mono mix; match when within +-8 % of the style BPM or half / double it.
usage (.venv-style4): style_fit_tiers.py --systems heartmula-swift-bf16-a,...  -> _out/style_fit_v2b.jsonl (E27 V2b copy of style_fit.py: any _out/<system> dir; the instrument is unchanged)"""
import argparse, json, re, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--systems", default="heartmula,acestep,yue2,minimax")
args = ap.parse_args()

import numpy as np, soundfile as sf, librosa, torch
from transformers import ClapModel, ClapProcessor

SRC = {"heartmula": C.OUT / "heartmula", "acestep": C.ACE_OUT / "acestep", "yue2": C.ACE_OUT / "yue2", "minimax": C.MM_OUT / "minimax"}
FEMALE = {"p01_pianopop", "p04_folk", "p05_synthpop", "p08_edm", "p09_jazz", "p11_zh_citypop"}
CLAP_SR, WIN_S, MIN_S = 48000, 10.0, 5.0

prompts = C.prompts(); ids = [p["id"] for p in prompts]
styles = [p["style"] for p in prompts]
bpm = {p["id"]: float(re.search(r"(\d+)\s*BPM", p["style"]).group(1)) for p in prompts}

clap_dir = C.ROOT / "_weights" / "larger_clap_music_and_speech"  # was larger_clap_music: degenerate checkpoint (MEASUREMENTS deviation 4)
model = ClapModel.from_pretrained(str(clap_dir)).eval()
proc = ClapProcessor.from_pretrained(str(clap_dir))


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def as_np(out):
    # transformers 5 may return a model-output object rather than a tensor
    if not torch.is_tensor(out):
        out = getattr(out, "pooler_output", None) if getattr(out, "pooler_output", None) is not None else out[0]
    return out.detach().float().numpy()


with torch.no_grad():
    t_in = proc(text=styles, return_tensors="pt", padding=True)
    T = l2(as_np(model.get_text_features(**t_in)))  # [12, D]


def song_embedding(wav_path):
    y, sr = librosa.load(str(wav_path), sr=CLAP_SR, mono=True)
    n = int(WIN_S * CLAP_SR)
    wins = [y[i:i + n] for i in range(0, len(y), n)]
    wins = [w for w in wins if len(w) >= MIN_S * CLAP_SR]
    embs = []
    with torch.no_grad():
        for i in range(0, len(wins), 8):
            a_in = proc(audios=wins[i:i + 8], sampling_rate=CLAP_SR, return_tensors="pt")
            embs.append(as_np(model.get_audio_features(**a_in)))
    E = l2(np.concatenate(embs, 0))
    return l2(E.mean(0)), y


def median_f0(vocals_path):
    v, sr = librosa.load(str(vocals_path), sr=16000, mono=True)
    f0, voiced, _ = librosa.pyin(v, fmin=65, fmax=1000, sr=sr, frame_length=1024)
    f = f0[voiced & np.isfinite(f0)]
    return float(np.median(f)) if len(f) else float("nan"), int(len(f))


def tempo_match(est, ref):
    return any(abs(est - ref * k) <= 0.08 * ref * k for k in (0.5, 1.0, 2.0))


out = C.OUT / "style_fit_v2b.jsonl"
done = set()
if out.exists():
    done = {(json.loads(l)["system"], json.loads(l)["id"]) for l in out.read_text().splitlines() if l.strip()}
for s in args.systems.split(","):
    SRC.setdefault(s, C.OUT / s)
    man = {json.loads(l)["id"]: json.loads(l) for l in (SRC[s] / "manifest.jsonl").read_text().splitlines() if l.strip()}
    for i, pid in enumerate(ids):
        if (s, pid) in done or not man.get(pid, {}).get("ok"):
            continue
        a, y = song_embedding(man[pid]["wav"])
        sims = T @ a
        order = list(np.argsort(-sims))
        rank = order.index(i) + 1
        margin = float(sims[i] - np.delete(sims, i).mean())
        f0, nvoiced = median_f0(SRC[s] / "_vocals" / f"{pid}.wav")
        tempo, _ = librosa.beat.beat_track(y=librosa.resample(y, orig_sr=CLAP_SR, target_sr=22050), sr=22050)
        tempo = float(np.atleast_1d(tempo)[0])
        row = dict(system=s, id=pid, clap_rank=rank, clap_top1=rank == 1, clap_margin=margin, clap_own=float(sims[i]),
                   clap_best=ids[order[0]], f0_median_hz=f0, f0_voiced_frames=nvoiced,
                   gender_expected="female" if pid in FEMALE else "male",
                   gender_read=("female" if f0 >= 165 else "male") if np.isfinite(f0) else "none",
                   tempo_est=tempo, tempo_ref=bpm[pid], tempo_match=tempo_match(tempo, bpm[pid]))
        row["gender_match"] = row["gender_read"] == row["gender_expected"]
        C.append_jsonl(out, row)
        print(f"{s:9s} {pid:15s} CLAP rank {rank:2d} (best {row['clap_best']}) margin {margin:+.3f} | F0 {f0:6.1f} Hz "
              f"{row['gender_read']}/{row['gender_expected']} | tempo {tempo:6.1f} vs {bpm[pid]:.0f} {'ok' if row['tempo_match'] else '--'}", flush=True)
print("DONE", flush=True)
