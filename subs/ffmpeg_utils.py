"""Small helpers around the ffmpeg / ffprobe command-line tools."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".ts"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class FFmpegError(RuntimeError):
    """Raised when ffmpeg / ffprobe fails or is missing."""


def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise FFmpegError(
            f"Not found on PATH: {', '.join(missing)}. "
            "Install FFmpeg (Windows: `winget install Gyan.FFmpeg`, macOS: `brew install ffmpeg`, "
            "Linux: `sudo apt install ffmpeg`) and open a new terminal."
        )


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        reason = lines[-1] if lines else f"exit code {proc.returncode}"
        raise FFmpegError(f"{cmd[0]} failed: {reason}")
    return proc


def probe_duration(path: Path) -> float:
    """Duration in seconds; raises FFmpegError if the file has no audio stream."""
    proc = run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_name:format=duration", "-of", "json", str(path),
    ])
    data = json.loads(proc.stdout or "{}")
    if not data.get("streams"):
        raise FFmpegError(f"No audio stream in {path.name}")
    duration = data.get("format", {}).get("duration")
    return float(duration) if duration not in (None, "N/A") else 0.0


def find_media_files(folder: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
