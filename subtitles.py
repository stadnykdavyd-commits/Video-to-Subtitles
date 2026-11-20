#!/usr/bin/env python3
"""Video -> subtitles (SRT / WebVTT).

Examples:
  python subtitles.py interview.mp4
  python subtitles.py videos/ --recursive --language en --format both
  python subtitles.py talk.mkv --model medium --clean -o my_subs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from subs.ffmpeg_utils import FFmpegError, find_media_files, require_tools
from subs.pipeline import process_file
from subs.transcribe import WHISPER_MODELS, FasterWhisperBackend, TranscriptionError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract audio, clean it, transcribe with Whisper and write subtitles.")
    p.add_argument("inputs", nargs="+", type=Path, help="video/audio files or folders")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output folder (default: 'output' next to each input; originals are never overwritten)")
    p.add_argument("-l", "--language", default=None, help="language code, e.g. en, de, uk, ru (default: auto-detect)")
    p.add_argument("-m", "--model", default="small", choices=WHISPER_MODELS,
                   help="Whisper model size: bigger = more accurate but slower (default: small)")
    p.add_argument("-f", "--format", default="srt", choices=["srt", "vtt", "both"], help="subtitle format")
    p.add_argument("--clean", action="store_true",
                   help="apply noise reduction + leveling before transcription (off by default: it did not "
                        "improve accuracy in our tests)")
    p.add_argument("--no-clean", action="store_true", help=argparse.SUPPRESS)  # kept for old command lines
    p.add_argument("--keep-audio", action="store_true", help="also save the cleaned 16 kHz WAV")
    p.add_argument("--max-chars", type=int, default=42, help="max characters per subtitle line (default: 42)")
    p.add_argument("--max-lines", type=int, default=2, help="max lines per subtitle (default: 2)")
    p.add_argument("-r", "--recursive", action="store_true", help="search folders recursively")
    return p


def collect(inputs: list[Path], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        if item.is_dir():
            files += find_media_files(item, recursive)
        elif item.is_file():
            files.append(item)
        else:
            print(f"! not found: {item}", file=sys.stderr)
    return files


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_tools()
        files = collect(args.inputs, args.recursive)
        if not files:
            print("No media files found.", file=sys.stderr)
            return 1
        backend = FasterWhisperBackend(args.model)
    except (FFmpegError, TranscriptionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    formats = ("srt", "vtt") if args.format == "both" else (args.format,)
    failed = 0
    for i, src in enumerate(files, 1):
        out_dir = args.output or src.parent / "output"
        print(f"[{i}/{len(files)}] {src.name} ...", flush=True)
        try:
            res = process_file(src, out_dir, backend, args.language, formats,
                               clean=args.clean and not args.no_clean, keep_audio=args.keep_audio,
                               max_chars=args.max_chars, max_lines=args.max_lines)
        except (FFmpegError, TranscriptionError) as exc:
            failed += 1
            print(f"    FAILED: {exc}", file=sys.stderr)
            continue
        print(f"    {res.cues} subtitles -> {', '.join(str(p) for p in res.outputs)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
