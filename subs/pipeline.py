"""Glue: one media file in -> subtitle file(s) out. Originals are never modified."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .audio import extract_audio
from .ffmpeg_utils import probe_duration
from .srt import Segment, refine, to_srt, to_vtt
from .transcribe import Backend


@dataclass
class Result:
    source: Path
    outputs: list[Path] = field(default_factory=list)
    cues: int = 0
    duration: float = 0.0


def process_file(
    src: Path,
    out_dir: Path,
    backend: Backend,
    language: str | None = None,
    formats: tuple[str, ...] = ("srt",),
    clean: bool = False,
    keep_audio: bool = False,
    max_chars: int = 42,
    max_lines: int = 2,
) -> Result:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(src)  # fails early with a clear message if there is no audio

    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(src, Path(tmp) / "audio.wav", clean=clean)
        raw: list[Segment] = backend.transcribe(wav, language)
        if keep_audio:
            kept = out_dir / f"{src.stem}.wav"
            kept.write_bytes(wav.read_bytes())

    cues = refine(raw, max_chars=max_chars, max_lines=max_lines)
    result = Result(source=src, cues=len(cues), duration=duration)
    for fmt in formats:
        path = out_dir / f"{src.stem}.{fmt}"
        text = to_srt(cues) if fmt == "srt" else to_vtt(cues)
        path.write_text(text, encoding="utf-8")
        result.outputs.append(path)
    return result
