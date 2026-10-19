# Acapella Rescore

Turn any song into an acapella-style backing: the original music is
removed, the voice is kept, and a hand-played percussion groove
(deep duff DUMs + claps) is laid underneath.

## The winning recipe

```bash
python rescore.py --mix song.mp3 --out song_acapella.wav \
  --perc-db 13 --backing haas-vocal --kit duff --groove lifted
```

- `--kit duff --groove lifted`: replays a transcribed reference groove
  (its phrases, fills, dynamics) on the song's own beat grid, bar-locked
  to the song's real drum hits so it can't drift.
- `--backing haas-vocal`: a Haas-widened double of the lead vocal fills
  the space where the music was.
- `--perc-db 13`: master beat level. The vocal is auto-leveled to a
  reference loudness first, so 13 means the same voice-vs-beats balance
  on every song.

## Beat levels (the product's tiers)

The lifted groove renders as two independent layers:

| Level | Sound | Flag |
|-------|-------|------|
| 1 | DUMs — deep duff bass strokes | `--layers dum` |
| 2 | Claps | `--layers dum,clap` |
| — | Audio-only (voice, no beats) | `--layers none` |

Each layer has its own loudness trim:

```bash
# DUMs only, a touch louder
python rescore.py --url "https://..." --out out.wav --layers dum --dum-db +3

# Full groove, claps pulled back a little
python rescore.py --mix song.mp3 --out out.wav --layers dum,clap --clap-db -3

# Just the voice, music removed, no beats
python rescore.py --mix song.mp3 --out out.wav --layers none
```

`--url` accepts a song link (YouTube etc.) and downloads it with yt-dlp.

New sounds (shaker, snaps, …) slot in as new layers the same way —
add a `*_hit()` synth, a layer buffer in `place_lifted_groove()`, and a
`--<name>-db` flag.

## Pipeline stages (for the web backend / credits)

1. **Fetch** (`--url`): download the song.
2. **Separate**: Demucs `mdx_q` splits vocals + drums (~minutes on CPU).
3. **Rescore**: beat-track, lift the groove, synthesize, mix (~seconds).
4. **Output**: WAV (convert to MP3 for delivery).

A credit = one completed run through stage 4.

## Files

- `rescore.py` — the pipeline.
- `models.py` — separator loading.
- `extract_groove.py` — transcribe a reference track's percussion into a
  reusable verse/chorus groove loop (`groove_ref.json`).
- `diag_groove.py` — forensic check: which groove hits land on real drum
  hits.
- `groove_ref.json` — the reference groove (verse/chorus loops).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch librosa scipy numpy soundfile yt-dlp
# + a Demucs-compatible separator (see models.py)
```
