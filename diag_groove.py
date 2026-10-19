"""Forensic check of the lifted groove against Insan's real drums.

For every placed groove hit, measures the distance to the nearest
transcribed Insan drum hit. Hits with nothing nearby 'should not be
there'; hits 50-150ms off are 'out of sync'.
"""
import json
import numpy as np
import librosa

import rescore as R
from models import load_stems_separator

MIX = "insan_half.mp3"
GROOVE_JSON = "groove_ref.json"
SR = 44100

print("separating ...", flush=True)
separate, info = load_stems_separator("mdx_q", stems=("vocals", "drums"),
                                      num_threads=2, single_submodel=True)
mix = R.load_stereo(MIX, SR)
stems = R.separate_long_stems(separate, mix, SR)
vocals = stems["vocals"].astype(np.float32)
drums = stems["drums"].astype(np.float32)
music = (mix - vocals).astype(np.float32)
drum_mono = librosa.to_mono(drums)
mono = librosa.to_mono(music)
dur = vocals.shape[1] / SR

print("beat grid ...", flush=True)
beats, tempo_bpm, snap_ms = R._track_grid(mono, SR, dur)
groove_src = drum_mono
auto_s = R._auto_shift(groove_src, SR, beats)
shift_eff = (4 - auto_s) % 4
bar_starts = np.array([beats[i] for i in range(len(beats))
                       if (i + shift_eff) % 4 == 0])
insan_bar = 4 * 60.0 / tempo_bpm
print(f"tempo {tempo_bpm:.1f}, {len(bar_starts)} bars", flush=True)

print("transcribing Insan drums ...", flush=True)
hits = R._transcribe_drums(drum_mono, SR)
print(f"{len(hits)} hits", flush=True)
insan_t = np.array(sorted(t for t, _, _ in hits))
insan_kick = np.array(sorted(t for t, k, _ in hits if k == "kick"))

with open(GROOVE_JSON) as f:
    gdata = json.load(f)

vmono = librosa.to_mono(vocals)
vtimes, vint = R._vocal_intensity_curve(vmono, SR)

# replay placement (mirror place_lifted_groove, capturing hit times)
n_loop = int(gdata["loop_bars"])
loops = {}
for name in ("verse", "chorus"):
    raw = [(float(p) % n_loop, str(c), float(v))
           for p, c, v in gdata.get(name, [])]
    best_s, best_e = 0, -1.0
    for s in range(n_loop):
        e = sum(v for p, c, v in raw if c == "dum" and (p - s) % 1.0 < 0.15)
        if e > best_e:
            best_e, best_s = e, s
    loops[name] = (raw, best_s)

allh = np.array(sorted(t for t, _, _ in hits))
placed = []  # (time, class)
for b, bs in enumerate(bar_starts):
    bs = float(bs)
    Ibar = float(np.interp(bs, vtimes, vint))
    name = "chorus" if Ibar > 0.5 else "verse"
    raw, shift = loops[name]
    lb = (b + shift) % n_loop
    rel = [((p - int(p)) * insan_bar, c, v) for p, c, v in raw
           if int(p) % n_loop == lb]
    rel = [(t, c, v) for t, c, v in rel
           if not (c in ("tak", "shk") and v < 0.5)]
    if not rel:
        continue
    best_dt = 0.0
    if len(allh):
        cand = np.array([bs + t for t, _, _ in rel])
        d0 = np.abs(allh[:, None] - cand[None, :]).min(axis=1)
        if int((d0 < 0.20).sum()) >= 2:
            best_n = -1
            for dt in np.arange(-0.15, 0.151, 0.01):
                d = np.abs(allh[:, None] - (cand + dt)[None, :]).min(axis=1)
                n = int((d < 0.06).sum())
                if n > best_n:
                    best_n, best_dt = n, float(dt)
    for t_rel, c, v in rel:
        placed.append((bs + t_rel + best_dt, c))

print(f"\nplaced {len(placed)} groove hits")
# distance of each placed hit to nearest Insan drum hit
bad = []
for t, c in placed:
    j = int(np.searchsorted(insan_t, t))
    d = 1e9
    if j < len(insan_t):
        d = min(d, abs(insan_t[j] - t))
    if j > 0:
        d = min(d, abs(insan_t[j - 1] - t))
    bad.append((d, t, c))
bad.sort(reverse=True)
print("\nworst 25 placed hits (dist ms | time | class):")
for d, t, c in bad[:25]:
    print(f"  {d*1000:6.0f}ms  @ {t:6.1f}s  {c}")
taks = [(d, t) for d, t, c in bad if c == "tak"]
print(f"\ntaks with no Insan hit within 150ms: "
      f"{sum(1 for d, _ in taks if d > 0.15)} of {len(taks)}")
print(f"taks 60-150ms off: {sum(1 for d, _ in taks if 0.06 < d <= 0.15)}")
dums = [(d, t) for d, t, c in bad if c == "dum"]
print(f"dums with no Insan hit within 150ms: "
      f"{sum(1 for d, _ in dums if d > 0.15)} of {len(dums)}")
