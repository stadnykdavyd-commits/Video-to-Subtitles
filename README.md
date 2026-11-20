# Video → Subtitles

Command-line tool that turns a video (or audio) file into ready-to-use **SRT / WebVTT** subtitles:

```
video ──► extract audio ──► clean up ──► transcribe (Whisper) ──► split & wrap ──► .srt / .vtt
```

- **Audio extraction** with FFmpeg: 16 kHz mono WAV, the format speech models expect
- **Optional audio cleanup** (`--clean`): high-pass / low-pass, FFT denoise, dynamic leveling. Off by default: in our tests it did not improve accuracy and on Russian speech made it worse
- **Transcription** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper): runs locally and offline, auto-detects the language, skips silence and music (VAD) to avoid invented subtitles
- **Readable cues**: long segments are split at sentence/comma boundaries, lines are balanced (max 42 characters × 2 lines by default), overlaps fixed, too-short cues extended
- **Batch mode**: files or whole folders, optional recursion
- Your originals are **never modified** — results go to an `output/` folder

Everything except the transcription step is pure Python standard library + FFmpeg. Whisper is an optional install.

## Install

1. Python 3.9+ and FFmpeg (Windows: `winget install Gyan.FFmpeg`)
2. For transcription: `pip install faster-whisper`
   (the model is downloaded once on first run, e.g. ~460 MB for `small`)

## Usage

```bash
python subtitles.py interview.mp4
python subtitles.py videos/ --recursive --language en --format both
python subtitles.py talk.mkv --model medium --clean -o my_subs
```

| Option | Meaning |
|---|---|
| `-o, --output` | output folder (default: `output/` next to each input) |
| `-l, --language` | language code (`en`, `de`, `uk`, `ru`…); default: auto-detect |
| `-m, --model` | `tiny` / `base` / `small` (default) / `medium` / `large-v3` |
| `-f, --format` | `srt` (default), `vtt` or `both` |
| `--clean` | apply noise reduction / leveling first (off by default) |
| `--keep-audio` | also save the cleaned 16 kHz WAV |
| `--max-chars`, `--max-lines` | subtitle layout (default 42 × 2) |
| `-r, --recursive` | search folders recursively |

Tip: for music, singing or rap use `--model large-v3`. Whisper can still invent a short phrase over an instrumental intro, so review the result. For plain speech the defaults are fine.

Tip: bigger models are more accurate but slower. `small` is a good balance on a normal laptop; use `medium` / `large-v3` for hard audio or rare languages.

## Measured accuracy

Word error rate (lower is better) on synthetic speech with known text, mixed with pink noise, auto-detected language:

| Model | English | Russian |
|---|---:|---:|
| small | 1.5% | 8.5% |
| large-v3 | 3.1% | 6.4% |

Differences of one or two words are noise at this sample size (about 65 words per test). The tests used Windows text-to-speech voices, not human speakers, so treat them as a pipeline check rather than a benchmark. On a 31-second recording of a real speaker (English, studio-quality voice, no reference transcript) `large-v3` produced 9 cues that read as clean, grammatical sentences with correct timing (45 s on a CPU-only laptop). On Ukrainian rap over a beat (the hardest case I tried, judged by ear, no reference lyrics) `large-v3` gave visibly more coherent lines than `small`.

## Tests

```bash
python -m unittest discover -s tests -v
```

17 tests: timestamp formatting, splitting/wrapping, overlap handling, SRT/VTT output, and end-to-end runs on a video generated with FFmpeg (a stand-in recognizer is used, so the tests need neither Whisper nor a model download).

## Project layout

```
subtitles.py        CLI
subs/ffmpeg_utils.py  ffmpeg/ffprobe helpers
subs/audio.py         extract + clean audio
subs/transcribe.py    Whisper backend (optional dependency)
subs/srt.py           split, wrap, SRT/VTT writer and parser
subs/pipeline.py      one file in -> subtitle files out
tests/
```

## License

MIT
