#!/usr/bin/env python3
"""rescore.py -- replace a song's instruments with an acapella arrangement.

Takes a mix (which it separates itself), beat-tracks the music stem, and
lays down beatbox-style vocal percussion on the groove plus a low human
hum following the chord roots, mixed back under the original vocal.
No instruments: the whole backing is voice-like sounds.

Usage:
    python rescore.py --mix clip.wav --out rescored.wav
    python rescore.py --vocals vox.wav --music mus.wav --out rescored.wav

Needs: librosa (pip install librosa)
"""

import argparse
import numpy as np
import soundfile as sf


def load_stereo(path: str, sr: int = 44100) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y.astype(np.float32)


def duff_hit(sr: int, gain: float = 1.0) -> np.ndarray:
    """Nasheed duff 'dum': deep and AUDIBLE -- fundamental 95->55Hz
    sits where speakers reproduce it, soft harmonics carry the weight
    to small speakers, smooth attack, tight enough to stay a hit."""
    from scipy import signal
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    freq = 55.0 + (95.0 - 55.0) * np.exp(-t * 22.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * np.exp(-t * 9.0)
    body = body + 0.30 * np.sin(2 * phase) * np.exp(-t * 11.0)
    atk = int(0.008 * sr)
    body[:atk] *= np.linspace(0.0, 1.0, atk) ** 0.5
    rng = np.random.default_rng(11)
    skin = rng.standard_normal(n) * np.exp(-t * 70.0) * 0.10
    sos = signal.butter(2, 350, btype="low", fs=sr, output="sos")
    skin = signal.sosfilt(sos, skin)
    x = body + skin
    return (x / max(np.abs(x).max(), 1e-6) * gain).astype(np.float32)


def clap_hit(sr: int, gain: float = 1.0, seed: int = 7) -> np.ndarray:
    """Hand clap: a few bandpassed noise bursts in quick succession."""
    from scipy import signal
    n = int(0.30 * sr)
    x = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for k in range(4):
        start = int(k * 0.011 * sr)
        m = min(int(0.025 * sr), n - start)
        env = np.exp(-np.arange(m) / (0.004 * sr))
        x[start:start + m] += rng.standard_normal(m) * env * (0.55 if k < 3 else 1.0)
    sos = signal.butter(4, [1100, 3200], btype="band", fs=sr, output="sos")
    x = signal.sosfilt(sos, x)
    x = x / max(np.abs(x).max(), 1e-6) * gain
    return x.astype(np.float32)


def snap_hit(sr: int, gain: float = 1.0, seed: int = 21) -> np.ndarray:
    """Finger snap: single bright, ultra-short noise burst."""
    from scipy import signal
    n = int(0.09 * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = rng.standard_normal(n) * np.exp(-t * 90.0)
    sos = signal.butter(4, 2500, btype="high", fs=sr, output="sos")
    x = signal.sosfilt(sos, x)
    return (x / max(np.abs(x).max(), 1e-6) * gain).astype(np.float32)


def tek_hit(sr: int, gain: float = 1.0, seed: int = 55) -> np.ndarray:
    """Duff 'tek': woody low-mid knock with an airy snap on top --
    modeled on the reference groove's taks. Seed varies the noise so
    repeated hits don't sound robotic."""
    from scipy import signal
    n = int(0.14 * sr)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = rng.standard_normal(n) * np.exp(-t * 60.0)
    sos = signal.butter(4, [250, 1000], btype="band", fs=sr, output="sos")
    x = signal.sosfilt(sos, x)
    x = x + 0.5 * np.sin(2 * np.pi * 220.0 * t) * np.exp(-t * 70.0)
    tick = rng.standard_normal(n) * np.exp(-t * 120.0) * 0.4
    sos2 = signal.butter(4, 2500, btype="high", fs=sr, output="sos")
    x = x + signal.sosfilt(sos2, tick)
    x = x / max(np.abs(x).max(), 1e-6) * gain
    return x.astype(np.float32)


def shaker_hit(sr: int, gain: float = 1.0) -> np.ndarray:
    """Shaker tick: short highpassed noise for offbeat groove."""
    from scipy import signal
    n = int(0.12 * sr)
    rng = np.random.default_rng(33)
    t = np.arange(n) / sr
    x = rng.standard_normal(n) * np.exp(-t * 55.0)
    sos = signal.butter(4, 5000, btype="high", fs=sr, output="sos")
    x = signal.sosfilt(sos, x)
    return (x / max(np.abs(x).max(), 1e-6) * gain).astype(np.float32)


def bbox_kick(sr: int, gain: float = 1.0) -> np.ndarray:
    """Beatbox 'boots': throatier pitch-dropping kick with a lip click."""
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    freq = 55.0 + (170.0 - 55.0) * np.exp(-t * 35.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * np.exp(-t * 26.0)
    rng = np.random.default_rng(11)
    click = rng.standard_normal(n) * np.exp(-t * 420.0) * 0.5
    x = body + click
    return (x / max(np.abs(x).max(), 1e-6) * gain).astype(np.float32)


def bbox_snare(sr: int, gain: float = 1.0) -> np.ndarray:
    """Beatbox 'pf': noisy mouth-snap with a low vocal-body thump."""
    from scipy import signal
    n = int(0.20 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    burst = rng.standard_normal(n) * np.exp(-t * 42.0)
    sos = signal.butter(4, [1300, 4600], btype="band", fs=sr, output="sos")
    burst = signal.sosfilt(sos, burst)
    body = np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 48.0) * 0.6
    x = burst * 0.85 + body
    return (x / max(np.abs(x).max(), 1e-6) * gain).astype(np.float32)


def bbox_hat(sr: int, gain: float = 1.0) -> np.ndarray:
    """Beatbox 'ts': ultra-short bright tongue tick."""
    from scipy import signal
    n = int(0.06 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(33)
    x = rng.standard_normal(n) * np.exp(-t * 130.0)
    sos = signal.butter(4, 6000, btype="high", fs=sr, output="sos")
    x = signal.sosfilt(sos, x)
    return (x / max(np.abs(x).max(), 1e-6) * gain).astype(np.float32)


def section_energy(music_mono: np.ndarray, sr: int,
                   beats: np.ndarray):
    """Split the song into bars and label loud bars as chorus.

    Returns (is_chorus, bounds): boolean per bar and the bar boundary
    times in seconds.
    """
    bounds = [float(beats[j]) for j in range(0, len(beats), 4)]
    bounds.append(len(music_mono) / sr)
    e = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        seg = music_mono[int(b0 * sr):int(b1 * sr)]
        e.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) + 1e-9)
    e = np.array(e)
    # chorus = the relatively biggest moments. If the song has real
    # dynamic contrast, that's bars clearly above the median; if it's
    # dynamically flat, fall back to the loudest third of bars so the
    # kit still breathes instead of going all-verse.
    med = np.median(e)
    if np.percentile(e, 90) > 1.25 * med:
        thr = 1.25 * med
    else:
        thr = np.percentile(e, 65)
    is_chorus = e >= thr
    # merge isolated bars so chorus regions are contiguous phrases
    # instead of scattered single bars (fewer, better riser entrances)
    from scipy.ndimage import binary_closing
    is_chorus = binary_closing(is_chorus, structure=np.ones(3))
    return is_chorus, bounds


def riser_swell(sr: int, dur_sec: float) -> np.ndarray:
    """Breathy 'tsshh': noise with an upward-sweeping bandpass and a
    crescendo, for building into a chorus."""
    from scipy import signal
    n = max(int(dur_sec * sr), 1)
    rng = np.random.default_rng(99)
    x = rng.standard_normal(n)
    chunks = 24
    out = np.zeros(n)
    zi = None
    for c in range(chunks):
        s = c * n // chunks
        e2 = (c + 1) * n // chunks
        f_lo = 700.0 * (6000.0 / 700.0) ** (c / chunks)
        sos = signal.butter(2, [f_lo, min(f_lo * 2.5, sr // 2 - 100)],
                            btype="band", fs=sr, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y, zi = signal.sosfilt(sos, x[s:e2], zi=zi)
        out[s:e2] = y
    out *= (np.arange(n) / n) ** 2.2  # crescendo
    return (out / max(np.abs(out).max(), 1e-6)).astype(np.float32)


def _hum_voice(f0: float, n: int, sr: int, seed: int) -> np.ndarray:
    """A single voice-like 'mmm': formant-shaped harmonics (the throat
    resonances that make a hum sound human instead of piano-like),
    vibrato that fades in like a real singer, a pitch scoop on the
    attack, breath noise, and gentle tremolo."""
    from scipy import signal
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    # pitch: scoop up into the note, vibrato fading in, slow drift
    scoop = 1.0 - 0.02 * np.exp(-t * 25.0)
    vib_depth = 0.007 * np.minimum(t / 0.45, 1.0)
    fmult = scoop * (1.0 + vib_depth * np.sin(2 * np.pi * 5.5 * t)
                     + 0.002 * np.sin(2 * np.pi * 0.6 * t + 1.3))
    ph = 2 * np.pi * np.cumsum(f0 * fmult) / sr
    # 'mmm' formants: strong low resonance, softer upper ones
    formants = ((280.0, 130.0, 1.0), (1050.0, 260.0, 0.5),
                (2400.0, 450.0, 0.20))
    x = np.zeros(n)
    h = 1
    while h <= 24:
        fh = f0 * h
        if fh >= sr / 2 - 100:
            break
        amp = h ** -1.25
        fg = sum(g * np.exp(-0.5 * ((fh - fc) / bw) ** 2)
                 for fc, bw, g in formants)
        x += amp * (0.2 + fg) * np.sin(h * ph + rng.uniform(0, 2 * np.pi))
        h += 1
    peak = np.abs(x).max()
    if peak > 1e-9:
        x /= peak
    # breath: bandpassed noise, like air through the nose
    sos = signal.butter(2, [500, 2800], btype='band', fs=sr, output='sos')
    bn = signal.sosfilt(sos, rng.standard_normal(n))
    bn *= 0.05 / (np.abs(bn).max() + 1e-9)
    x = x + bn
    # tremolo in sync with the vibrato
    x *= 1.0 + 0.07 * np.sin(2 * np.pi * 5.5 * t + 0.8)
    return x


def hum_layer(n_samples: int, sr: int, music_mono: np.ndarray,
              beats: np.ndarray, hum_db: float = -12.0,
              vocal_peak: float = 1.0) -> np.ndarray:
    """A low human 'mmm' following the chord roots, bar by bar.

    Reads the chord root of each bar from the music stem's chroma,
    then sings it an octave or two down with vibrato and a soft
    attack -- like the bass singer in an acapella group.
    """
    import librosa
    from scipy import signal
    hop = 2048
    chroma = librosa.feature.chroma_stft(y=music_mono, sr=sr,
                                         hop_length=hop, n_fft=4096)
    ctimes = np.arange(chroma.shape[1]) * hop / sr
    bounds = [float(beats[j]) for j in range(0, len(beats), 4)]
    bounds.append(n_samples / sr)
    roots = []
    thirds = []  # semitones above root for the harmony voice (3 or 4)
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        m = (ctimes >= b0) & (ctimes < b1)
        if m.sum() == 0:
            roots.append(roots[-1] if roots else 9)
            thirds.append(thirds[-1] if thirds else 4)
            continue
        cm = chroma[:, m].mean(axis=1)
        root = int(np.argmax(cm))
        semi = 4 if cm[(root + 4) % 12] >= cm[(root + 3) % 12] else 3
        roots.append(root)
        thirds.append(semi)
    # pitch class -> frequency in the 55-110 Hz bass-hum range (A1 = pc 9)
    freqs = [55.0 * 2.0 ** ((pc - 9) / 12.0) for pc in roots]
    freqs = [f * 2.0 if f < 55.0 else f for f in freqs]
    # harmony voice: a third above, sung an octave up for clarity
    harm_freqs = [f * 2.0 * 2.0 ** (s / 12.0)
                  for f, s in zip(freqs, thirds)]

    hum = np.zeros(n_samples, dtype=np.float64)
    atk = int(0.12 * sr)
    overlap = int(0.18 * sr)
    for idx, (b0, b1, f0, fh) in enumerate(zip(bounds[:-1], bounds[1:],
                                              freqs, harm_freqs)):
        i0 = int(b0 * sr)
        i1 = min(int(b1 * sr) + overlap, n_samples)
        if i1 <= i0:
            continue
        seg_n = i1 - i0
        # bass voice + harmony voice, both sung from the throat
        x = (_hum_voice(f0, seg_n, sr, seed=1000 + idx)
             + 0.5 * _hum_voice(fh, seg_n, sr, seed=2000 + idx))
        a = min(atk, seg_n)
        x[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
        r = min(int(0.25 * sr), seg_n)
        x[-r:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(r) / r)
        hum[i0:i1] += x
    # "mmm": keep it dark and chesty, but let the upper formants through
    sos = signal.butter(4, 3200, btype="low", fs=sr, output="sos")
    hum = signal.sosfilt(sos, hum)
    peak = np.abs(hum).max()
    if peak > 1e-6:
        hum = hum / peak * vocal_peak * 10.0 ** (hum_db / 20.0)
    return hum.astype(np.float32)


def simple_reverb(x: np.ndarray, sr: int, rt60: float = 0.9,
                  wet: float = 0.22) -> np.ndarray:
    """Cheap room: decaying-noise impulse convolved per channel."""
    from scipy import signal
    n_ir = int(rt60 * sr)
    rng = np.random.default_rng(3)
    t = np.arange(n_ir) / sr
    ir = rng.standard_normal((2, n_ir)) * np.exp(-6.9 * t / rt60)
    sos = signal.butter(2, 4000, btype="low", fs=sr, output="sos")
    ir = signal.sosfilt(sos, ir)
    ir = ir / max(np.abs(ir).max(), 1e-6)
    tail = np.stack([signal.fftconvolve(x[c], ir[c], mode="full")[:x.shape[1]]
                     for c in range(2)])
    tail = tail / max(np.abs(tail).max(), 1e-6) * np.abs(x).max()
    return ((1.0 - wet) * x + wet * tail).astype(np.float32)


def _pan_add(L: np.ndarray, R: np.ndarray, hit: np.ndarray, idx: int,
             pan: float, gain: float, n_samples: int) -> None:
    end = min(idx + len(hit), n_samples)
    if idx >= n_samples or end <= idx:
        return
    h = hit[:end - idx] * gain
    gl = np.cos((pan + 1.0) * np.pi / 4.0)  # constant-power pan
    gr = np.sin((pan + 1.0) * np.pi / 4.0)
    L[idx:end] += h * gl
    R[idx:end] += h * gr


def place_hits(n_samples: int, sr: int, beat_times: np.ndarray,
               shift: int = 0, kit: str = "beatbox",
               intensity: np.ndarray | None = None) -> np.ndarray:
    """Vocal percussion: beatbox kick on 1 & 3 (center), mouth-snare on
    2 & 4 (alternating sides), tongue-tick hats on offbeats, with slight
    humanization and a 16th-note fill every 8th bar. `intensity` is a
    per-beat 0/1 array: quiet bars get hats only (verse feel), loud bars
    get the full kit (chorus). Returns dry stereo; reverb is applied to
    the full backing bus in main()."""
    L = np.zeros(n_samples, dtype=np.float32)
    R = np.zeros(n_samples, dtype=np.float32)
    if kit == "beatbox":
        kick, snare, hat = bbox_kick(sr), bbox_snare(sr), bbox_hat(sr)
    elif kit == "duff":
        # dum / tek / soft dum: deep frame-drum groove, no tsk anywhere
        kick, snare, hat = duff_hit(sr), tek_hit(sr), duff_hit(sr, gain=0.35)
    else:  # classic drum-machine kit
        kick, snare, hat = duff_hit(sr), clap_hit(sr), shaker_hit(sr)
    clap = clap_hit(sr)
    snap = snap_hit(sr)
    rng = np.random.default_rng(42)
    n_beats = len(beat_times)
    for i, bt in enumerate(beat_times):
        pos = (i + shift) % 4
        bar = i // 4
        full = intensity is None or float(intensity[i]) > 0.5
        jitter = int(rng.uniform(-0.006, 0.006) * sr)  # +/-6 ms feel
        v = 10 ** rng.uniform(-1.5, 1.5) / 20.0         # +/-1.5 dB
        idx = int(bt * sr) + jitter
        if idx < 0:
            continue
        # fill: last beat of every 8th bar becomes a 16th-note roll
        if bar % 8 == 7 and pos == 3 and i + 1 < n_beats and full:
            step = (beat_times[i + 1] - bt) / 4.0
            for k, h in enumerate((kick, hat, snare, hat)):
                _pan_add(L, R, h, int((bt + k * step) * sr) + jitter,
                         0.0, (0.9 if k % 2 == 0 else 0.7) * v, n_samples)
            continue
        if pos in (0, 2):      # downbeats: kick, centered (beat 1 heavier)
            kgain = (1.0 if pos == 0 else 0.85) * v if full else 0.5 * v
            _pan_add(L, R, kick, idx, 0.0, kgain, n_samples)
        else:                  # backbeats: snare, alternating sides
            if full:
                pan = -0.35 if (i // 2) % 2 == 0 else 0.35
                _pan_add(L, R, snare, idx, pan, 0.9 * v, n_samples)
                if kit == "beatbox":
                    # hand claps + finger snaps layered over the mouth snare
                    _pan_add(L, R, clap, idx, pan, 0.5 * v, n_samples)
                    _pan_add(L, R, snap, idx + int(0.004 * sr), -pan,
                             0.4 * v, n_samples)
                elif kit == "classic":
                    _pan_add(L, R, snap, idx + int(0.004 * sr), -pan,
                             0.7 * v, n_samples)
                # duff: pure dum/tek, nothing layered on top
        if i + 1 < n_beats:  # hat tick on the offbeat (always, softer solo)
            off = int((bt + beat_times[i + 1]) / 2 * sr)
            _pan_add(L, R, hat, off, 0.15 * (-1.0 if i % 2 else 1.0),
                     0.45 * v if full else 0.35 * v, n_samples)
    perc = np.stack([np.tanh(L * 1.2), np.tanh(R * 1.2)]).astype(np.float32)
    return perc


def place_nasheed_duff(n_samples: int, sr: int,
                       beat_times: np.ndarray, shift: int = 0) -> np.ndarray:
    """Arabic frame-drum groove like nasheed percussion: deep DUM on 1
    and 3, crisp tek on 2 and 4, a feather-light ka on the offbeat after
    2 and 4. Sparse and hypnotic -- deliberately not a drum-kit pattern.
    Returns dry stereo; reverb is applied to the full backing bus."""
    L = np.zeros(n_samples, dtype=np.float32)
    R = np.zeros(n_samples, dtype=np.float32)
    dum = duff_hit(sr)
    tek = tek_hit(sr)
    ka = tek_hit(sr, gain=0.3)
    rng = np.random.default_rng(7)
    n_beats = len(beat_times)
    for i, bt in enumerate(beat_times):
        pos = (i + shift) % 4
        jitter = int(rng.uniform(-0.008, 0.008) * sr)  # hand-played feel
        v = 10 ** rng.uniform(-1.0, 1.0) / 20.0
        idx = int(bt * sr) + jitter
        if idx < 0 or idx >= n_samples:
            continue
        if pos == 0:  # DUM
            _pan_add(L, R, dum, idx, 0.0, 1.0 * v, n_samples)
        elif pos == 2:  # dum, a touch softer
            _pan_add(L, R, dum, idx, 0.0, 0.85 * v, n_samples)
        else:  # tek, alternating sides
            pan = -0.25 if (i // 2) % 2 == 0 else 0.25
            _pan_add(L, R, tek, idx, pan, 0.8 * v, n_samples)
            if i + 1 < n_beats:  # feather-light ka on the offbeat
                off = int((bt + beat_times[i + 1]) / 2 * sr) + jitter
                _pan_add(L, R, ka, off, -pan * 0.4, 0.5 * v, n_samples)
    perc = np.stack([np.tanh(L * 1.2), np.tanh(R * 1.2)]).astype(np.float32)
    return perc


def place_nasheed_transcribed(n_samples: int, sr: int,
                              hits: list, beats: np.ndarray,
                              shift: int = 0) -> np.ndarray:
    """Sparse nasheed duff with the original's exact timing. Each bar
    keeps only its structural hits -- the strongest kick near beats 1
    and 3 becomes DUM, the strongest snare near 2 and 4 becomes tek --
    placed at the hit's own timestamp, so the groove can't sit off-beat
    the way a rigid grid can. Hats and ghost hits are dropped. Bars
    with no transcribed kick fall back to a soft grid DUM. Returns dry
    stereo; reverb is applied to the full backing bus."""
    L = np.zeros(n_samples, dtype=np.float32)
    R = np.zeros(n_samples, dtype=np.float32)
    dum = duff_hit(sr)
    tek = tek_hit(sr)
    rng = np.random.default_rng(7)
    n_beats = len(beats)
    if n_beats < 4:
        return np.stack([L, R]).astype(np.float32)
    iv = float(np.median(np.diff(beats)))
    kicks = [(t, v) for t, k, v in hits if k == "kick"]
    snares = [(t, v) for t, k, v in hits if k == "snare"]
    alt = 0
    for i in range(n_beats):
        rpos = (i + shift) % 4
        lo, hi = beats[i] - 0.3 * iv, beats[i] + 0.3 * iv
        if rpos in (0, 2):  # downbeats: DUM
            cand = [(t, v) for t, v in kicks if lo <= t <= hi]
            if cand:
                t, v = max(cand, key=lambda c: c[1])
                g = 0.85 + 0.15 * v
                if rpos == 2:
                    g *= 0.9
                _pan_add(L, R, dum, int(t * sr), 0.0, g, n_samples)
            else:  # no drum to copy: soft grid DUM keeps the pulse
                j = int(rng.uniform(-0.008, 0.008) * sr)
                idx = int(beats[i] * sr) + j
                _pan_add(L, R, dum, idx, 0.0,
                         0.5 if rpos == 0 else 0.4, n_samples)
        else:  # backbeats: tek
            cand = [(t, v) for t, v in snares if lo <= t <= hi]
            if cand:
                alt += 1
                t, v = max(cand, key=lambda c: c[1])
                pan = -0.25 if alt % 2 == 0 else 0.25
                _pan_add(L, R, tek, int(t * sr), pan, 0.8 * v,
                         n_samples)
            # no forced tek: sparse is the point
    perc = np.stack([np.tanh(L * 1.2), np.tanh(R * 1.2)]).astype(np.float32)
    return perc


def _vocal_intensity_curve(vocal_mono: np.ndarray, sr: int):
    """Vocal intensity 0..1 sampled every 0.1 s: RMS in 0.4 s windows,
    normalized by the 90th percentile, smoothed over ~1.5 s. Lets the
    groove breathe with the singer instead of looping constantly."""
    import librosa
    hop = int(sr * 0.1)
    frame = int(sr * 0.4)
    rms = librosa.feature.rms(y=np.ascontiguousarray(vocal_mono),
                              frame_length=frame,
                              hop_length=hop)[0]
    times = np.arange(len(rms)) * hop / sr
    peak = float(np.percentile(rms, 90))
    inten = np.clip(rms / max(peak, 1e-6), 0.0, 1.0)
    k = 15
    ker = np.ones(k) / k
    inten = np.convolve(inten, ker, mode="same")
    return times, inten


def _fast_pulse_grid(drum_mono: np.ndarray, sr: int, dur: float,
                     hits: list, pulse: str = "quarter"):
    """Pulse grid for the dum-dum-tak groove. The DP tempo is settled to
    the quarter-note song feel (60-170 BPM); pulse='quarter' keeps the
    groove at the voice's own tempo, 'eighth' doubles it. The pattern
    phase is searched directly against the transcribed kick hits (ground
    truth): the phase where the most strong kicks land on the pattern's
    DUM pulses wins. Returns (pulse_times, pulse_tempo, zero_idx,
    bar_pulses)."""
    import librosa
    tempo, dbeats = librosa.beat.beat_track(y=drum_mono, sr=sr,
                                            units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    while tempo > 170:
        tempo /= 2.0
    while tempo < 60:
        tempo *= 2.0
    pulse_tempo = tempo if pulse == "quarter" else tempo * 2.0
    bar_pulses = 4 if pulse == "quarter" else 8
    dum_offsets = [0] if pulse == "quarter" else [0, 4]
    period = 60.0 / pulse_tempo
    bar = bar_pulses * period
    kicks = np.array(sorted(t for t, k, v in hits
                            if k == "kick" and v > 0.5))
    best_ph, best_score = 0.0, -np.inf
    if len(kicks) >= 4:
        for ph in np.arange(0, bar, 0.005):
            starts = np.arange(ph, dur, bar)
            dums = np.concatenate([starts + o * period
                                   for o in dum_offsets])
            dums = dums[dums < dur]
            j = np.searchsorted(kicks, dums)
            d = np.minimum(
                np.abs(kicks[np.clip(j, 0, len(kicks) - 1)] - dums),
                np.abs(kicks[np.clip(j - 1, 0, len(kicks) - 1)] - dums))
            score = float((d < 0.06).sum())
            if score > best_score:
                best_score, best_ph = score, float(ph)
    else:  # no kicks to lock to: fall back to the DP beat phase
        dbeats = np.atleast_1d(dbeats).astype(float)
        if len(dbeats):
            ph = np.mod(dbeats, period)
            ang = 2 * np.pi * ph / period
            best_ph = float((np.angle(np.exp(1j * ang).mean())
                             / (2 * np.pi) * period) % period)
    grid = np.arange(best_ph, dur, period)
    while len(grid) and grid[0] - period >= 0:
        grid = np.concatenate([[grid[0] - period], grid])
    zero_idx = int(np.argmin(np.abs(grid - best_ph)))
    return grid, pulse_tempo, zero_idx, bar_pulses


def place_dum_dum_tak_locked(n_samples: int, sr: int, pulses: np.ndarray,
                             period: float, hits: list,
                             zero_idx: int, vtimes: np.ndarray,
                             vint: np.ndarray, pulse: str = "quarter",
                             bar_pulses: int = 4,
                             groove: str = "dense") -> np.ndarray:
    """Dum-dum-tak with per-bar timing lock AND vocal following. Each
    8-pulse bar is slid up to +/-150 ms to best fit the transcribed drum
    hits, then every pattern slot snaps to the nearest real hit
    (kick->DUM, snare->TAK) within 80 ms; slots with no hit nearby play
    softly on the grid. The vocal intensity curve shapes the groove:
    a soft heartbeat while the singer is silent, a sparser DUM-TAK when
    he holds back, the full pattern when he belts. Returns dry stereo;
    reverb is applied to the full backing bus in main()."""
    L = np.zeros(n_samples, dtype=np.float32)
    R = np.zeros(n_samples, dtype=np.float32)
    dum = duff_hit(sr)
    tek = tek_hit(sr)
    # pattern position -> (sound, gain, pan, hit family to snap to).
    # groove "dum": just DUM on every bar_pulses beats, louder.
    # quarter pulse: DUM(1) dum(2) TAK(3), at the voice's own tempo.
    # eighth pulse:  DUM(1) dum(&) TAK(2) DUM(3) dum(&) TAK(4).
    if groove == "dum":
        pattern = {0: (dum, 1.15, 0.0, "kick")}
    elif pulse == "quarter":
        pattern = {0: (dum, 1.00, 0.0, "kick"),
                   1: (dum, 0.80, 0.0, "kick"),
                   2: (tek, 0.90, 0.15, "snare")}
    else:
        pattern = {0: (dum, 1.00, 0.0, "kick"),
                   1: (dum, 0.70, 0.0, "kick"),
                   2: (tek, 0.85, 0.18, "snare"),
                   4: (dum, 0.92, 0.0, "kick"),
                   5: (dum, 0.70, 0.0, "kick"),
                   6: (tek, 0.85, -0.18, "snare")}
    slot_idx = sorted(pattern.keys())
    kicks = np.array(sorted(t for t, k, v in hits if k == "kick"))
    snares = np.array(sorted(t for t, k, v in hits if k == "snare"))
    allh = np.array(sorted(t for t, _, _ in hits))
    rng = np.random.default_rng(9)
    n_bars = (len(pulses) - zero_idx) // bar_pulses
    for b in range(max(n_bars, 0)):
        base_i = zero_idx + b * bar_pulses
        slots = np.array([pulses[base_i + s] for s in slot_idx])
        best_dt = 0.0
        if len(allh):
            d0 = np.abs(allh[:, None] - slots[None, :]).min(axis=1)
            if int((d0 < 0.15).sum()) >= 2:
                best_n = -1
                for dt in np.arange(-0.15, 0.151, 0.01):
                    d = np.abs(allh[:, None]
                               - (slots + dt)[None, :]).min(axis=1)
                    n = int((d < 0.06).sum())
                    if n > best_n:
                        best_n, best_dt = n, float(dt)
        for s, st in zip(slot_idx, slots + best_dt):
            snd, g, pan, want = pattern[s]
            I = float(np.interp(st, vtimes, vint))  # vocal intensity
            if I < 0.12:
                if s != 0:
                    continue  # singer silent: heartbeat DUM only
                t_hit, g_hit = float(st), 0.35
            else:
                m = 0.45 + 0.55 * I
                if pulse == "eighth" and s in (1, 5) and I < 0.30:
                    continue  # soft singing: sparser DUM-TAK
                pool = kicks if want == "kick" else snares
                t_hit, g_hit = float(st), g * 0.5 * m
                if len(pool):
                    j = int(np.searchsorted(pool, st))
                    cand = []
                    if j < len(pool):
                        cand.append(pool[j])
                    if j > 0:
                        cand.append(pool[j - 1])
                    t0 = min(cand, key=lambda t: abs(t - st))
                    if abs(t0 - st) <= 0.08:
                        t_hit, g_hit = float(t0), g * m
            jitter = int(rng.uniform(-0.006, 0.006) * sr)
            v = 10 ** (rng.uniform(-1.0, 1.0) / 20.0)
            _pan_add(L, R, snd, int(t_hit * sr) + jitter, pan,
                     g_hit * v, n_samples)
    # head/tail pulses outside full bars: soft grid keeps the pulse,
    # still following the singer
    covered = set(range(zero_idx, zero_idx + max(n_bars, 0) * bar_pulses))
    for i, pt in enumerate(pulses):
        if i in covered:
            continue
        s = (i - zero_idx) % bar_pulses
        slot = pattern.get(s)
        if slot is None:
            continue
        I = float(np.interp(pt, vtimes, vint))
        if I < 0.12 and s != 0:
            continue
        snd, g, pan, _ = slot
        _pan_add(L, R, snd, int(pt * sr), pan,
                 g * 0.4 * (0.35 + 0.65 * I), n_samples)
    perc = np.stack([np.tanh(L * 1.2), np.tanh(R * 1.2)]).astype(np.float32)
    return perc


def place_lifted_groove(n_samples: int, sr: int, bar_starts: np.ndarray,
                       insan_bar: float, gdata: dict,
                       vtimes: np.ndarray, vint: np.ndarray,
                       hits: list) -> dict:
    """Replay a transcribed reference groove (see extract_groove.py).
    The reference's hit pattern -- its phrases, variations, fills, the
    beat melody -- is mapped bar-for-bar onto this song's bar grid, so
    the groove keeps its musical shape at the song's own tempo. Each
    bar is micro-aligned to the song's real drum hits (ground truth),
    so accents can't land off-beat. Verse / chorus loops switch with
    the vocal intensity; the groove keeps playing softly even when the
    singer is silent. Returns a dict of dry stereo layers:
    {"dum": deep bass strokes, "clap": claps/snaps}. Reverb is applied
    to the full backing bus in main(). Layers let the product offer
    DUMs and claps as separately-priced/controllable beat levels."""
    L = np.zeros(n_samples, dtype=np.float32)
    R = np.zeros(n_samples, dtype=np.float32)
    dum = duff_hit(sr)
    # claps instead of wood knocks (the Titanium v2 recipe):
    # a small bank so no two hits sound identical
    tak_bank = [clap_hit(sr, seed=s) for s in (71, 72, 73, 74)]
    ka = tek_hit(sr, gain=0.35)
    sounds = {"dum": (dum, 1.15, 0.0), "tak": (None, 0.65, None),
              "shk": (ka, 0.8, None)}
    # separate accumulators per beat layer (the product's level 1/2)
    Ld = np.zeros(n_samples, dtype=np.float32)
    Rd = np.zeros(n_samples, dtype=np.float32)
    Lc = np.zeros(n_samples, dtype=np.float32)
    Rc = np.zeros(n_samples, dtype=np.float32)
    n_loop = int(gdata["loop_bars"])
    loops = {}
    for name in ("verse", "chorus"):
        raw = [(float(p) % n_loop, str(c), float(v))
               for p, c, v in gdata.get(name, [])]
        # rotate the loop so its dum-heaviest downbeat sits on bar 0
        best_s, best_e = 0, -1.0
        for s in range(n_loop):
            e = sum(v for p, c, v in raw
                    if c == "dum" and (p - s) % 1.0 < 0.15)
            if e > best_e:
                best_e, best_s = e, s
        loops[name] = (raw, best_s)
    rng = np.random.default_rng(11)
    tak_alt = 0
    allh = np.array(sorted(t for t, _, _ in hits))
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
        # micro-align this bar to the song's real drum hits
        best_dt = 0.0
        if len(allh):
            cand = np.array([bs + t for t, _, _ in rel])
            d0 = np.abs(allh[:, None] - cand[None, :]).min(axis=1)
            if int((d0 < 0.20).sum()) >= 2:
                best_n = -1
                for dt in np.arange(-0.15, 0.151, 0.01):
                    d = np.abs(allh[:, None]
                               - (cand + dt)[None, :]).min(axis=1)
                    n = int((d < 0.06).sum())
                    if n > best_n:
                        best_n, best_dt = n, float(dt)
        # Insan's drum hits around this bar: the ground truth
        pool = [(t, k) for t, k, vv in hits
                if bs - 0.25 <= t <= bs + insan_bar + 0.25]
        if len(pool) >= 3:
            # snap every groove hit onto a real Insan hit (greedy,
            # loudest first); groove hits with no Insan hit nearby are
            # dropped -- they are the ones that sound out of sync or
            # like they should not be there
            order = sorted(range(len(rel)), key=lambda i: -rel[i][2])
            used, play = set(), []
            for i in order:
                t_rel, c, v = rel[i]
                tg = bs + t_rel + best_dt
                bj, bd = -1, 0.12
                for j, (th, kh) in enumerate(pool):
                    if j in used:
                        continue
                    d = abs(th - tg)
                    if d < bd:
                        bd, bj = d, j
                if bj >= 0:
                    used.add(bj)
                    play.append((pool[bj][0], c, v))
        else:
            # intro/break: no drums to snap to, groove plays as-is
            play = [(bs + t_rel + best_dt, c, v) for t_rel, c, v in rel]
        for t, c, v in play:
            if t < 0 or t * sr >= n_samples:
                continue
            snd, g0, pan0 = sounds.get(c, (tak_bank[0], 0.9, None))
            if c == "tak":
                # a different knock each time, like a real hand
                snd = tak_bank[int(rng.integers(0, len(tak_bank)))]
            if pan0 is None:
                tak_alt += 1
                pan0 = 0.15 if tak_alt % 2 else -0.15
            I = float(np.interp(t, vtimes, vint))
            m = 0.35 if I < 0.12 else (0.6 + 0.4 * I)
            jitter = int(rng.uniform(-0.004, 0.004) * sr)
            if c == "dum":
                _pan_add(Ld, Rd, snd, int(t * sr) + jitter, pan0,
                         v * g0 * m, n_samples)
            else:
                _pan_add(Lc, Rc, snd, int(t * sr) + jitter, pan0,
                         v * g0 * m, n_samples)
    layers = {
        "dum": np.stack([np.tanh(Ld * 1.2),
                         np.tanh(Rd * 1.2)]).astype(np.float32),
        "clap": np.stack([np.tanh(Lc * 1.2),
                          np.tanh(Rc * 1.2)]).astype(np.float32),
    }
    return layers


def _transcribe_drums(drums_mono: np.ndarray, sr: int):
    """Detect every drum hit in the drums stem and classify it as
    kick/snare/hat by brightness of the transient. Returns a list of
    (time_sec, kind, vel). Replaying these with beatbox sounds copies the
    original groove exactly, so it can never drift off-beat."""
    import librosa
    hop = 256
    odf = librosa.onset.onset_strength(y=drums_mono, sr=sr,
                                       hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(odf)), sr=sr,
                                   hop_length=hop)
    med = float(np.median(odf))
    rng = float(odf.max() - med)
    thresh = med + 0.8 * (float(odf.std()) or 1e-9)
    hits, last_t = [], -1.0
    win = int(0.09 * sr)
    for i in range(1, len(odf) - 1):
        if not (odf[i] > thresh and odf[i] >= odf[i - 1]
                and odf[i] > odf[i + 1]):
            continue
        t = float(times[i])
        if t - last_t < 0.045:  # min spacing: no double triggers
            continue
        s0 = int(t * sr)
        seg = drums_mono[s0:min(len(drums_mono), s0 + win)]
        if len(seg) < 64:
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        cent = float((spec * freqs).sum() / (spec.sum() + 1e-9))
        kind = ("kick" if cent < 500 else
                "snare" if cent < 2600 else "hat")
        vel = 0.35 + 0.65 * float(np.clip((odf[i] - med) / (rng + 1e-9),
                                         0.0, 1.0))
        hits.append((t, kind, vel))
        last_t = t
    return hits


def place_transcribed(n_samples: int, sr: int,
                      hits: list, kit: str,
                      is_chorus: np.ndarray,
                      bar_bounds: np.ndarray,
                      beats: np.ndarray) -> np.ndarray:
    """Vocal percussion from a transcribed drum pattern: every original
    hit is replayed as a beatbox sound at its exact timestamp. Chorus
    bars get the full pattern; verse bars keep kick + hats only, softer.
    Instrumental stretches with no transcribed hits (intros, breakdowns)
    get a soft groove filled in so they never go empty. Returns dry
    stereo; reverb is applied to the full backing bus."""
    L = np.zeros(n_samples, dtype=np.float32)
    R = np.zeros(n_samples, dtype=np.float32)
    if kit == "beatbox":
        sounds = {"kick": bbox_kick(sr), "snare": bbox_snare(sr),
                  "hat": bbox_hat(sr)}
    elif kit == "duff":
        # dum / tek / soft dum: deep frame-drum groove, no tsk anywhere
        sounds = {"kick": duff_hit(sr), "snare": tek_hit(sr),
                  "hat": duff_hit(sr, gain=0.4)}
    else:
        sounds = {"kick": duff_hit(sr), "snare": clap_hit(sr),
                  "hat": shaker_hit(sr)}
    snap = snap_hit(sr)
    bars = np.searchsorted(bar_bounds, [h[0] for h in hits],
                           side="right") - 1
    bars = np.clip(bars, 0, len(is_chorus) - 1)
    alt = 0
    for (t, kind, vel), b in zip(hits, bars):
        full = bool(is_chorus[b])
        if not full and kind == "snare":
            continue  # verse feel: kick + hats only
        g = vel * (1.0 if full else 0.8)
        idx = int(t * sr)
        if idx >= n_samples:
            continue
        if kind == "kick":
            _pan_add(L, R, sounds["kick"], idx, 0.0, g, n_samples)
        elif kind == "snare":
            alt += 1
            pan = -0.35 if alt % 2 == 0 else 0.35
            _pan_add(L, R, sounds["snare"], idx, pan, 0.9 * g,
                     n_samples)
            if kit == "beatbox":
                _pan_add(L, R, snap, idx + int(0.004 * sr), -pan,
                         0.4 * g, n_samples)
        else:
            pan = 0.15 * (-1.0 if alt % 2 else 1.0)
            _pan_add(L, R, sounds["hat"], idx, pan, 0.5 * g, n_samples)
    # --- gap fill: instrumental stretches (intros, breakdowns) with no
    # transcribed hits get a soft groove so they never go empty. Only
    # fills runs of 3+ consecutive hit-less beats.
    ht = np.array(sorted(t for t, _, _ in hits))
    iv = float(np.median(np.diff(beats))) if len(beats) >= 2 else 0.5
    empty = []
    for bt in beats:
        d_near = 1e9
        if len(ht):
            j = int(np.searchsorted(ht, bt))
            if j < len(ht):
                d_near = min(d_near, abs(ht[j] - bt))
            if j > 0:
                d_near = min(d_near, abs(ht[j - 1] - bt))
        empty.append(d_near > 0.6 * iv)
    n_b, i = len(beats), 0
    while i < n_b:
        if not empty[i]:
            i += 1
            continue
        j = i
        while j < n_b and empty[j]:
            j += 1
        if j - i >= 3:
            for k in range(i, j):
                idx = int(beats[k] * sr)
                _pan_add(L, R, sounds["kick"], idx, 0.0, 0.45, n_samples)
                if k + 1 < n_b:
                    off = int((beats[k] + beats[k + 1]) / 2 * sr)
                    _pan_add(L, R, sounds["hat"], off, 0.0, 0.20,
                             n_samples)
        i = j
    perc = np.stack([np.tanh(L * 1.2), np.tanh(R * 1.2)]).astype(np.float32)
    return perc


def haas_widen(src_mono: np.ndarray, sr: int, delay_ms: float = 25.0,
               gain_db: float = 0.0) -> np.ndarray:
    """Haas effect: mono source -> wide stereo, left dry and right
    delayed by ~15-35 ms. The brain reads it as one wide voice rather
    than an echo. gain_db is applied to both sides."""
    g = 10 ** (gain_db / 20.0)
    d = int(delay_ms * sr / 1000.0)
    L = src_mono.astype(np.float32) * g
    R = np.zeros_like(L)
    if d < len(L):
        R[d:] = L[:-d] if d > 0 else L
    return np.stack([L, R]).astype(np.float32)


def _refine_tempo(src_mono: np.ndarray, sr: int, dur: float,
                  hint_bpm: float):
    """Comb-filter tempo search: tries multiples of the tracker's hint and
    fine steps around them, scoring each tempo by how much onset energy
    lands on its grid (best phase kept). The DP tracker alone can sit ~2%
    off, which drifts audibly over a full song. Returns (bpm, phase)."""
    import librosa
    D = np.abs(librosa.stft(src_mono, n_fft=2048, hop_length=512))
    _, P = librosa.decompose.hpss(D)
    onset = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(P, ref=np.max), sr=sr, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(onset)), sr=sr,
                                   hop_length=512)

    def score_bpm(bpm):
        # on-beats minus off-beats: the true tempo puts onsets ON the
        # grid and quiet between; half-tempo would put strong beats
        # on the "off" grid too and scores ~zero.
        period = 60.0 / bpm
        best_e, best_ph = -np.inf, 0.0
        for ph in np.linspace(0, period, 48):
            grid = np.arange(ph, dur, period)
            idx = np.searchsorted(times, grid).clip(0, len(onset) - 1)
            off = np.searchsorted(times, grid + period / 2).clip(
                0, len(onset) - 1)
            e = (float(onset[idx].mean() - onset[off].mean())
                 if len(idx) else -np.inf)
            if e > best_e:
                best_e, best_ph = e, ph
        return best_e, best_ph

    cands = {hint_bpm * m for m in (0.5, 0.75, 1.0, 1.5, 2.0)}
    cands = {c for c in cands if 40.0 <= c <= 220.0}
    scored = [(score_bpm(c)[0], c) for c in cands]
    _, best_bpm = max(scored)
    fine = [best_bpm + d for d in np.arange(-6.0, 6.01, 0.5)]
    fine = [f for f in fine if 40.0 <= f <= 220.0]
    best_e, best_ph, best_b = -np.inf, 0.0, best_bpm
    for f in fine:
        e, ph = score_bpm(f)
        if e > best_e:
            best_e, best_ph, best_b = e, ph, f
    return best_b, best_ph


def _track_grid(src_mono: np.ndarray, sr: int, dur: float):
    """Beat-track one source and fill a rigid tempo grid across the whole
    track. The DP tracker gives a tempo hint and micro-timing; a
    comb-filter search pins the exact tempo and phase. Returns
    (beats, tempo_bpm, snap_err_ms). Raises SystemExit if nothing steady."""
    import librosa
    tempo, beats = librosa.beat.beat_track(y=src_mono, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.atleast_1d(beats).astype(float)
    # octave correction: beat trackers often lock onto double-time
    while tempo > 150 and len(beats) >= 8:
        tempo /= 2.0
        beats = beats[::2]
    if len(beats) < 4:
        raise SystemExit("no steady beat")
    bpm, phase = _refine_tempo(src_mono, sr, dur, tempo)
    period = 60.0 / bpm
    # The DP tracker often drops out in sparse sections (breakdowns,
    # outros), leaving the percussion silent there. Fill a rigid grid at
    # the refined tempo across the whole track, snapping to detected
    # beats where they exist. Extended back to 0 so ambient intros still
    # get hum bars and gap-fill groove positions.
    start = phase
    while start - period >= 0:
        start -= period
    grid = np.arange(start, dur, period)
    grid = grid[grid >= 0]
    filled = []
    for g in grid:
        j = int(np.searchsorted(beats, g))
        cands = []
        if j < len(beats):
            cands.append(beats[j])
        if j > 0:
            cands.append(beats[j - 1])
        c = min(cands, key=lambda b: abs(b - g))
        filled.append(c if abs(c - g) <= 0.06 else g)
    filled = np.asarray(filled)
    j = np.searchsorted(filled, beats)
    d = np.minimum(np.abs(filled[np.clip(j, 0, len(filled) - 1)] - beats),
                   np.abs(filled[np.clip(j - 1, 0, len(filled) - 1)] - beats))
    snap_ms = float(np.median(d)) * 1000.0
    return filled, bpm, snap_ms


def _auto_shift(groove_mono: np.ndarray, sr: int, beats: np.ndarray) -> int:
    """Find which beat position (0-3) is the true downbeat by listening
    for kick thumps: the rotation with the most low-end energy on beats
    1 & 3 wins. Fixes patterns that start on the wrong beat, which is
    what makes syncopated songs feel 'messed up'."""
    from scipy import signal
    sos = signal.butter(4, 150, btype="low", fs=sr, output="sos")
    kick = np.abs(signal.sosfilt(sos, groove_mono))
    k = max(int(0.05 * sr), 1)
    kick = np.convolve(kick, np.ones(k) / k, mode="same")
    n = len(kick)
    e = np.array([kick[int(b * sr)] if 0 <= int(b * sr) < n else 0.0
                  for b in beats])
    best_s, best_score = 0, -np.inf
    for s in range(4):
        down = e[s::4].mean() if len(e[s::4]) else 0.0
        down = (down + (e[(s + 2) % 4::4].mean() if len(e[(s + 2) % 4::4]) else 0.0)) / 2
        up_idx = [i for i in range(4) if i not in (s, (s + 2) % 4)]
        up = np.mean([e[i::4].mean() for i in up_idx if len(e[i::4])])
        score = down - up
        if score > best_score:
            best_score, best_s = score, s
    return best_s


def separate_long(separate, mix: np.ndarray, sr: int,
                  chunk_sec: float = 30.0, overlap_sec: float = 2.0
                  ) -> np.ndarray:
    """Separate a long mix in overlapping chunks with crossfade stitching.

    The separator is built for short windows; feeding it minutes of audio
    at once silently dies, so chunk it like the live pipeline does.
    """
    n = mix.shape[1]
    chunk = int(chunk_sec * sr)
    ov = int(overlap_sec * sr)
    step = chunk - ov
    out = np.zeros_like(mix)
    weight = np.zeros(n, dtype=np.float32)
    w = np.ones(chunk, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32)
    w[:ov] = ramp
    w[-ov:] = 1.0 - ramp
    for start in range(0, n, step):
        end = min(start + chunk, n)
        v = separate(mix[:, start:end])
        L = end - start
        out[:, start:end] += v[:, :L] * w[:L]
        weight[start:end] += w[:L]
        print(f"  separated {start / sr:.0f}-{end / sr:.0f}s", flush=True)
    return (out / np.maximum(weight, 1e-6)).astype(np.float32)


def separate_long_stems(separate, mix: np.ndarray, sr: int,
                        chunk_sec: float = 30.0, overlap_sec: float = 2.0
                        ) -> dict:
    """Chunked separation returning {stem_name: (C, N)} in a single pass,
    with the same overlap-crossfade stitching as separate_long."""
    n = mix.shape[1]
    chunk = int(chunk_sec * sr)
    ov = int(overlap_sec * sr)
    step = chunk - ov
    first = separate(mix[:, :min(chunk, n)])
    stems = {k: np.zeros_like(mix) for k in first}
    weight = np.zeros(n, dtype=np.float32)
    w = np.ones(chunk, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32)
    w[:ov] = ramp
    w[-ov:] = 1.0 - ramp
    for start in range(0, n, step):
        end = min(start + chunk, n)
        out = first if start == 0 else separate(mix[:, start:end])
        L = end - start
        for k, v in out.items():
            stems[k][:, start:end] += v[:, :L] * w[:L]
        weight[start:end] += w[:L]
        print(f"  separated {start / sr:.0f}-{end / sr:.0f}s", flush=True)
    return {k: (v / np.maximum(weight, 1e-6)).astype(np.float32)
            for k, v in stems.items()}


def spectral_gate(x: np.ndarray, sr: int, alpha: float = 3.0,
                  smooth: int = 5) -> np.ndarray:
    """Light stationary-noise cleanup on a (C, N) stem.

    Estimates the noise floor per frequency bin from the quietest frames
    and softly gates anything near it. Cheap enough to run on a laptop.
    """
    import librosa
    from scipy.ndimage import uniform_filter1d
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        X = librosa.stft(x[c], n_fft=2048, hop_length=512)
        mag, ph = np.abs(X), np.exp(1j * np.angle(X))
        noise = np.percentile(mag, 20, axis=1, keepdims=True)
        thr = np.maximum(noise * alpha, 1e-6 * mag.max())
        mask = np.clip((mag - thr) / np.maximum(mag, 1e-8), 0.0, 1.0)
        mask = uniform_filter1d(mask, size=smooth, axis=1)
        y = librosa.istft(mask * mag * ph, hop_length=512, length=x.shape[1])
        out[c] = y.astype(np.float32)
    return out


def silence_gate(x: np.ndarray, sr: int, thresh_db: float = -40.0,
                 floor_db: float = -35.0, frame_ms: float = 20.0,
                 smooth_ms: float = 120.0) -> np.ndarray:
    """Duck near-silent frames hard to kill separation warble between phrases.

    Frames whose level sits more than thresh_db below the stem's peak are
    faded down to floor_db (smoothed to avoid chattering).
    """
    fl = max(1, int(frame_ms * sr / 1000))
    n = x.shape[1]
    nf = int(np.ceil(n / fl))
    xp = np.pad(x, ((0, 0), (0, nf * fl - n)))
    frames = xp.reshape(x.shape[0], nf, fl)
    rms = np.sqrt((frames ** 2).mean(axis=2)).max(axis=0)
    db = 20 * np.log10(rms / max(rms.max(), 1e-9) + 1e-12)
    gate = np.ones(nf, dtype=np.float32)
    gate[db < thresh_db] = 10 ** (floor_db / 20.0)
    k = max(1, int(smooth_ms / frame_ms))
    gate = np.convolve(gate, np.ones(k) / k, mode="same").astype(np.float32)
    env = np.repeat(gate, fl)[:n]
    return (x * env).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mix", help="full mix; dialogue/music separated internally")
    g.add_argument("--vocals", help="dialogue stem (use with --music)")
    g.add_argument("--url", help="link to a song (YouTube etc.); downloaded "
                                 "with yt-dlp, then processed as --mix")
    ap.add_argument("--music", help="music stem (use with --vocals)")
    ap.add_argument("--out", required=True, help="output wav path")
    ap.add_argument("--perc-db", type=float, default=13.0,
                    help="master beat level in dB (default: 13, the winning "
                         "recipe)")
    ap.add_argument("--dum-db", type=float, default=0.0,
                    help="level-1 DUM loudness trim in dB, added to --perc-db "
                         "(default: 0)")
    ap.add_argument("--clap-db", type=float, default=0.0,
                    help="level-2 clap loudness trim in dB, added to --perc-db "
                         "(default: 0)")
    ap.add_argument("--layers", default="dum,clap",
                    help="beat layers in the mix, comma-separated: dum "
                         "(level 1, deep duff bass), clap (level 2). "
                         "'none' = audio-only vocals, no beats "
                         "(default: dum,clap)")
    ap.add_argument("--denoise", action="store_true", default=True,
                    help="light noise cleanup on the dialogue/vocal stem "
                         "(default: on)")
    ap.add_argument("--no-denoise", dest="denoise", action="store_false")
    ap.add_argument("--shift-beats", type=int, default=0,
                    help="rotate the kick/snare pattern by N beats if the "
                         "downbeat feels off")
    ap.add_argument("--wet", type=float, default=0.22,
                    help="reverb wet amount on the backing bus "
                         "(default: 0.22)")
    ap.add_argument("--hum-db", type=float, default=-12.0,
                    help="hum level in dB relative to the vocal "
                         "(negative = under the vocal)")
    ap.add_argument("--no-hum", dest="hum", action="store_false",
                    help="skip the bass hum layer")
    ap.add_argument("--backing", default="hum",
                    choices=["hum", "haas-vocal", "haas-hum", "haas-both"],
                    help="what fills the backing behind the beat: synth hums "
                         "(default), a Haas-widened double of the lead "
                         "vocal, Haas-widened hum notes, or both")
    ap.add_argument("--kit", default="beatbox",
                    choices=["beatbox", "classic", "duff"],
                    help="percussion kit: beatbox (mouth sounds), classic "
                         "(drum machine), duff (deep dum/tek frame drum, "
                         "no tsk)")
    ap.add_argument("--groove", default="sparse",
                    choices=["sparse", "dense", "dum", "lifted"],
                    help="duff kit only: sparse = nasheed frame-drum groove "
                         "(DUM on 1&3, tek on 2&4); dense = dum-dum-tak "
                         "wedding groove; dum = just DUM on every Nth beat "
                         "(see --dum-every); lifted = replay a transcribed "
                         "reference groove (see --groove-file)")
    ap.add_argument("--groove-file", default="groove_ref.json",
                    help="with --groove lifted: JSON from extract_groove.py")
    ap.add_argument("--dum-every", type=int, default=1,
                    help="with --groove dum: DUM every N beats (2 = wider, "
                         "slower pulse)")
    ap.add_argument("--pulse", default="quarter",
                    choices=["quarter", "eighth"],
                    help="groove pulse: quarter = dum-dum-tak at the voice's "
                         "own tempo (default); eighth = double-time")
    ap.add_argument("--full-kit", action="store_true", default=False,
                    help="full kit on every bar (v2 feel); default follows "
                         "the verse/chorus dynamics (sparser verses)")
    ap.add_argument("--no-transcribe", dest="transcribe",
                    action="store_false", default=True,
                    help="force the classic grid pattern instead of "
                         "replaying the song's own transcribed drum groove")
    ap.add_argument("--model", default="mdx_q")
    ap.add_argument("--single-submodel", action="store_true", default=True)
    ap.add_argument("--no-single-submodel", dest="single_submodel",
                    action="store_false")
    args = ap.parse_args()
    if args.vocals and not args.music:
        ap.error("--vocals needs --music")

    sr = 44100
    mix_path = args.mix
    if args.url:
        # the product's link-in flow: fetch the song, then treat it as --mix
        import subprocess, tempfile, os, sys
        print(f"downloading {args.url} ...", flush=True)
        tmpd = tempfile.mkdtemp(prefix="acapella-url-")
        mix_path = os.path.join(tmpd, "song.mp3")
        # Use the same interpreter's yt-dlp module (bare `yt-dlp` is not
        # guaranteed on PATH inside a venv), and the android player client,
        # which YouTube's bot check usually lets through on datacenter IPs.
        subprocess.run([sys.executable, "-m", "yt_dlp", args.url,
                        "--no-check-certificate",
                        "--extractor-args", "youtube:player_client=android",
                        "--retries", "5", "-x", "--audio-format", "mp3",
                        "-o", mix_path], check=True)
        print(f"  saved to {mix_path}", flush=True)
    if mix_path:
        print("separating dialogue/music ...", flush=True)
        from models import load_stems_separator
        separate, info = load_stems_separator(
            args.model, stems=("vocals", "drums"),
            num_threads=2, single_submodel=args.single_submodel)
        mix = load_stereo(mix_path, sr)
        stems = separate_long_stems(separate, mix, sr)
        vocals = stems["vocals"].astype(np.float32)
        drums = stems["drums"].astype(np.float32)
        music = (mix - vocals).astype(np.float32)
    else:
        vocals = load_stereo(args.vocals, sr)
        music = load_stereo(args.music, sr)
        n = min(vocals.shape[1], music.shape[1])
        vocals, music = vocals[:, :n], music[:, :n]
        drums = None

    if args.denoise:
        print("cleaning up the vocal stem ...", flush=True)
        vocals = silence_gate(spectral_gate(vocals, sr), sr)

    # level the voice to a consistent loudness on every song, so that
    # --perc-db always means the same voice-vs-beats balance. Target is
    # the Insan vocal level (the "Love it" reference).
    vrms = float(np.sqrt((vocals.astype(np.float64) ** 2).mean()))
    if vrms > 1e-4:
        vg = 0.135 / vrms
        # don't go crazy on near-silent stems; cap at +18 dB
        vg = min(vg, 8.0)
        vocals = (vocals * vg).astype(np.float32)
        print(f"  vocal level: {20*np.log10(vg):+.1f} dB to match reference",
              flush=True)

    print("beat-tracking (drums vs music, keeping the tighter lock) ...",
          flush=True)
    import librosa
    mono = librosa.to_mono(music)  # chords/arrangement read the full music
    dur = vocals.shape[1] / sr
    cands = []
    if drums is not None and np.abs(drums).mean() > 1e-4:
        cands.append(("drums", librosa.to_mono(drums)))
    cands.append(("music", mono))
    best = None
    for name, src in cands:
        try:
            res = _track_grid(src, sr, dur)
        except SystemExit:
            continue
        print(f"  {name}: ~{res[1]:.0f} BPM, snap err {res[2]:.0f} ms",
              flush=True)
        if best is None or res[2] < best[1][2]:
            best = (name, res)
    if best is None:
        raise SystemExit("couldn't find a steady beat in the music stem; "
                         "this works best on rhythmic music.")
    name, (beats, tempo_bpm, snap_ms) = best
    print(f"  using the {name} stem", flush=True)
    # rotate the pattern so the kick lands on the true downbeat
    groove = librosa.to_mono(drums) if drums is not None else mono
    auto_s = _auto_shift(groove, sr, beats)
    shift_eff = ((4 - auto_s) % 4 + args.shift_beats) % 4
    print(f"  downbeat at beat position {auto_s + 1}", flush=True)
    print(f"  tempo ~{tempo_bpm:.0f} BPM, {len(beats)} beats "
          f"(grid-filled, snap err {snap_ms:.0f} ms)", flush=True)

    n_samples = vocals.shape[1]
    beats = beats[beats * sr < n_samples]
    vpeak = float(np.abs(vocals).max())
    print("reading the arrangement (verse/chorus) ...", flush=True)
    is_chorus, bar_bounds = section_energy(mono, sr, beats)
    intensity = np.repeat(is_chorus.astype(float), 4)[:len(beats)]
    n_chorus = int(is_chorus.sum())
    print(f"  {n_chorus}/{len(is_chorus)} bars sound like chorus", flush=True)
    print(f"laying down the {args.kit} kit ...", flush=True)
    drum_mono = (librosa.to_mono(drums) if drums is not None
                 and np.abs(drums).mean() > 1e-4 else None)
    hits = (_transcribe_drums(drum_mono, sr)
            if (drum_mono is not None and args.transcribe) else [])
    if args.kit == "duff":
        if args.groove == "lifted" and drum_mono is not None:
            # replay the reference's transcribed groove (its phrases,
            # fills, dynamics) on the song's own beat grid -- the
            # comb-refined tempo and downbeat, bar-locked to the real
            # drum hits so accents land on the beat
            import json
            with open(args.groove_file) as f:
                gdata = json.load(f)
            bar_starts = np.array([beats[i] for i in range(len(beats))
                                   if (i + shift_eff) % 4 == 0])
            insan_bar = 4 * 60.0 / tempo_bpm
            vtimes, vint = _vocal_intensity_curve(librosa.to_mono(vocals),
                                                  sr)
            print(f"  lifting groove from {args.groove_file} "
                  f"@ ~{tempo_bpm:.0f} BPM, {len(bar_starts)} bars",
                  flush=True)
            perc_layers = place_lifted_groove(n_samples, sr, bar_starts,
                                              insan_bar, gdata, vtimes,
                                              vint, hits)
            groove_desc = "lifted reference groove"
        elif len(hits) >= 20 and args.groove in ("dense", "dum"):
            # duff groove at the voice's own tempo, each bar re-locked to
            # the song's real drum hits so it can't drift, and breathing
            # with the vocal intensity curve
            pulses, ptempo, zero_idx, bar_pulses = _fast_pulse_grid(
                drum_mono, sr, dur, hits, pulse=args.pulse)
            if args.groove == "dum":
                bar_pulses = max(1, args.dum_every)  # DUM every N beats
                gname = "DUM pulse"
            else:
                gname = "dum-dum-tak"
            vtimes, vint = _vocal_intensity_curve(librosa.to_mono(vocals),
                                                  sr)
            print(f"  groove pulse ~{ptempo:.0f} BPM, {gname} "
                  f"({args.pulse}, bar-locked, vocal-following)", flush=True)
            perc = place_dum_dum_tak_locked(
                n_samples, sr, pulses, 60.0 / ptempo, hits, zero_idx,
                vtimes, vint, pulse=args.pulse, bar_pulses=bar_pulses,
                groove=args.groove)
            groove_desc = (f"{gname} groove ({args.pulse}, bar-locked, "
                           f"vocal-following)")
        elif len(hits) >= 20:
            # nasheed groove from the song's own drum hits: exact timing,
            # sparse DUM/tek voicing -- never a drum-kit pattern
            print(f"  transcribed {len(hits)} drum hits from the original",
                  flush=True)
            perc = place_nasheed_transcribed(n_samples, sr, hits, beats,
                                             shift=shift_eff)
            groove_desc = "nasheed duff groove"
        else:
            print("  playing the nasheed duff groove ...", flush=True)
            perc = place_nasheed_duff(n_samples, sr, beats,
                                      shift=shift_eff)
            groove_desc = "nasheed duff groove"
    elif len(hits) >= 20:
        # replay the song's own drum pattern as beatbox: exact groove,
        # zero drift
        print(f"  transcribed {len(hits)} drum hits from the original",
              flush=True)
        perc = place_transcribed(n_samples, sr, hits, args.kit,
                                 is_chorus, bar_bounds, beats)
        groove_desc = "transcribed original groove"
    else:
        if args.full_kit:
            intensity = None  # v2 feel: full kit on every bar
        perc = place_hits(n_samples, sr, beats, shift=shift_eff,
                          kit=args.kit, intensity=intensity)
        groove_desc = "grid pattern"
        perc_layers = None
    if groove_desc == "lifted reference groove":
        # product beat levels: DUMs (level 1, the deep duff bass stroke)
        # and claps (level 2) get independent loudness; --layers picks
        # which are in the mix ("none" = audio-only vocals, no beats)
        want = set(args.layers.split(",")) if args.layers else set()
        dum_g = 10 ** ((args.perc_db + args.dum_db) / 20.0)
        clap_g = 10 ** ((args.perc_db + args.clap_db) / 20.0)
        parts = []
        if "dum" in want and "dum" in perc_layers:
            parts.append(perc_layers["dum"] * dum_g)
            print(f"  beat level 1: DUMs @ {args.perc_db + args.dum_db:+.0f} dB",
                  flush=True)
        if "clap" in want and "clap" in perc_layers:
            parts.append(perc_layers["clap"] * clap_g)
            print(f"  beat level 2: claps @ {args.perc_db + args.clap_db:+.0f} dB",
                  flush=True)
        if not parts:
            print("  audio-only: no beat layers", flush=True)
        perc = (sum(parts) if parts
                else np.zeros((2, n_samples), dtype=np.float32))
    else:
        perc *= 10 ** (args.perc_db / 20.0)
    bus = perc
    if args.hum:
        if args.backing in ("hum", "haas-hum", "haas-both"):
            print("singing the bass + harmony hums ...", flush=True)
            hum = hum_layer(n_samples, sr, mono, beats, hum_db=args.hum_db,
                            vocal_peak=vpeak)
            if args.backing == "haas-hum":
                print("  widening the hums with the Haas effect ...",
                      flush=True)
                bus = bus + haas_widen(hum, sr, delay_ms=22.0)
            else:
                bus = bus + np.stack([hum, hum])
        if args.backing in ("haas-vocal", "haas-both"):
            print("doubling the lead vocal with the Haas effect ...",
                  flush=True)
            bus = bus + haas_widen(librosa.to_mono(vocals), sr,
                                   delay_ms=25.0, gain_db=args.hum_db)
    # breathy riser swelling into each chorus entrance
    n_risers = 0
    for b in range(1, len(is_chorus)):
        if is_chorus[b] and not is_chorus[b - 1]:
            dur_r = bar_bounds[b] - bar_bounds[b - 1]
            r = riser_swell(sr, dur_r)
            idx = int(bar_bounds[b - 1] * sr)
            end = min(idx + len(r), n_samples)
            if end > idx:
                bus[0, idx:end] += r[:end - idx] * 0.5
                bus[1, idx:end] += r[:end - idx] * 0.5
                n_risers += 1
    if n_risers:
        print(f"  {n_risers} riser(s) into the chorus", flush=True)
    bus = simple_reverb(bus, sr, wet=args.wet)

    out = np.tanh(vocals + bus).astype(np.float32)
    sf.write(args.out, out.T, sr)
    peak = float(np.abs(out).max())
    print(f"wrote {args.out} ({n_samples / sr:.1f}s, peak {peak:.2f})",
          flush=True)
    backing_desc = {"hum": f" + bass hum ({args.hum_db:+.0f} dB vs vocal)",
                    "haas-vocal": " + Haas vocal double",
                    "haas-hum": " + Haas-widened hums",
                    "haas-both": " + Haas vocal double + Haas hums"}[args.backing]
    print(f"backing: {args.kit} kit ({groove_desc})"
          + (backing_desc if args.hum else "")
          + " (--shift-beats N to rotate the pattern)", flush=True)


if __name__ == "__main__":
    main()
