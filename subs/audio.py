"""Step 1-2: pull the audio out of a video and (optionally) clean it for speech recognition."""

from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import run

# Speech-friendly cleanup chain:
#  - high-pass 80 Hz removes rumble / handling noise
#  - low-pass 8 kHz drops hiss above the speech band (Whisper works at 16 kHz anyway)
#  - afftdn is a gentle FFT denoiser
#  - dynaudnorm evens out quiet and loud speakers so quiet words aren't missed
CLEAN_FILTER = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,dynaudnorm=f=150:g=15"


def extract_audio(src: Path, dst: Path, clean: bool = True) -> Path:
    """Write a 16 kHz mono 16-bit WAV (the format speech models expect). Never touches `src`."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
    if clean:
        cmd += ["-af", CLEAN_FILTER]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    run(cmd)
    return dst
