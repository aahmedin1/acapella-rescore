"""Lift the drum groove out of a reference track.

Transcribes the reference's percussion (via HPSS) into a loop of hits:
each hit is (position_in_loop_bars, class, vel) with class in
{dum, tak, shk}. Two loops are extracted -- a sparser 'verse' loop and
a hotter 'chorus' loop -- so the replayed beat has phrases and dynamics,
a beat melody, not a metronome.

Usage:
    python extract_groove.py ref.mp3 groove.json
"""
import json
import sys

import numpy as np
import librosa
import scipy.signal as sig


def band_rms(seg, sr, lo, hi):
    sos = sig.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    b = sig.sosfilt(sos, seg)
    return float(np.sqrt((b ** 2).mean()))


def transcribe_loop(yp, sr, start, n_bars, bar):
    """Transcribe one loop window, quantized to 8th notes per voice.
    Raw onset detection catches transient smear as machine-gun clusters
    (e.g. hits at 58/116/232ms for one drum stroke); quantizing each
    voice (dum/tak/shk) to the 8th-note grid and keeping only the
    loudest hit per slot keeps the groove's musical skeleton -- its
    phrases and accents -- without the bouncing-ball smear.
    Returns list of (pos_in_loop_bars, class, vel)."""
    seg = yp[int(start * sr): int((start + n_bars * bar) * sr)]
    hop = 256
    odf = librosa.onset.onset_strength(y=np.ascontiguousarray(seg), sr=sr,
                                       hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(odf)), sr=sr,
                                   hop_length=hop)
    med = float(np.median(odf))
    thresh = med + 0.9 * float(odf.std() or 1e-9)
    raw, last_t = [], -1.0
    win = int(0.06 * sr)
    for i in range(1, len(odf) - 1):
        if not (odf[i] > thresh and odf[i] >= odf[i - 1]
                and odf[i] > odf[i + 1]):
            continue
        t = float(times[i])
        if t - last_t < 0.04:
            continue
        s0 = int(t * sr)
        w = seg[s0:min(len(seg), s0 + win)]
        if len(w) < 32:
            continue
        e_low = band_rms(w, sr, 40, 200)
        e_mid = band_rms(w, sr, 800, 4000)
        e_hi = band_rms(w, sr, 5500, 11000)
        cls = ("dum", "tak", "shk")[int(np.argmax([e_low, e_mid, e_hi]))]
        vel = float(np.clip(np.sqrt((w ** 2).mean()) * 6.0, 0.25, 1.0))
        raw.append((t, cls, vel))
        last_t = t
    # quantize: one slot per 8th note per voice, keep the loudest
    eighth = bar / 8.0
    best = {}
    for t, cls, vel in raw:
        slot = (cls, int(round(t / eighth)))
        if slot not in best or vel > best[slot][2]:
            best[slot] = (t, cls, vel)
    hits = [(t / bar, cls, round(vel, 3)) for t, cls, vel in best.values()]
    hits.sort(key=lambda h: h[0])
    return hits


def main():
    ref_path, out_path = sys.argv[1], sys.argv[2]
    y, sr = librosa.load(ref_path, sr=22050, mono=True)
    print("separating percussion ...", flush=True)
    yp, _ = librosa.effects.hpss(y)
    tempo, beats = librosa.beat.beat_track(y=yp, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    bar = 4 * 60.0 / tempo
    dur = len(y) / sr
    print(f"reference tempo ~{tempo:.1f} BPM, bar {bar:.3f}s", flush=True)

    # onset density per 8-bar window -> pick a sparse and a hot loop
    hop = 512
    odf = librosa.onset.onset_strength(y=np.ascontiguousarray(yp), sr=sr,
                                       hop_length=hop)
    ot = librosa.frames_to_time(np.arange(len(odf)), sr=sr, hop_length=hop)
    n_bars, step = 8, 2  # 8-bar loops, 2-bar step
    cands = []
    b0 = float(beats[0])
    t = b0
    while t + n_bars * bar < dur - 5:
        m = (ot >= t) & (ot < t + n_bars * bar)
        cands.append((t, float(odf[m].sum())))
        t += step * bar
    cands.sort(key=lambda c: c[1])
    verse_t = cands[len(cands) // 4][0]   # sparse-ish, not the emptiest
    chorus_t = cands[-1][0]               # hottest window
    print(f"verse loop @ {verse_t:.1f}s, chorus loop @ {chorus_t:.1f}s",
          flush=True)

    # snap loop starts to the nearest detected beat
    beats = np.atleast_1d(beats).astype(float)
    verse_t = float(beats[np.argmin(np.abs(beats - verse_t))])
    chorus_t = float(beats[np.argmin(np.abs(beats - chorus_t))])

    verse_hits = transcribe_loop(yp, sr, verse_t, n_bars, bar)
    chorus_hits = transcribe_loop(yp, sr, chorus_t, n_bars, bar)
    print(f"verse: {len(verse_hits)} hits, chorus: {len(chorus_hits)} hits",
          flush=True)
    data = {"ref_tempo": tempo, "ref_bar": bar, "loop_bars": n_bars,
            "verse": verse_hits, "chorus": chorus_hits,
            "source": ref_path}
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
