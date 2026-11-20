"""Step 3: speech -> text. Whisper (via faster-whisper) is an OPTIONAL dependency.

Everything else in this project works without it; the backend is a tiny interface so
another engine (or a test double) can be plugged in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .srt import Segment

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")


class TranscriptionError(RuntimeError):
    pass


class Backend(Protocol):
    def transcribe(self, wav: Path, language: str | None) -> list[Segment]: ...


class FasterWhisperBackend:
    """Local, offline speech recognition. The model is downloaded once on first use."""

    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "auto"):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise TranscriptionError(
                "faster-whisper is not installed. Run:  pip install faster-whisper\n"
                "(it is optional: only the transcription step needs it)"
            ) from exc
        self._WhisperModel = WhisperModel
        self._name = model
        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def _run(self, wav: Path, language: str | None) -> list[Segment]:
        segments, _info = self._model.transcribe(
            str(wav),
            language=language,         # None = auto-detect
            vad_filter=True,           # skip silence / music: fewer hallucinated subtitles
            beam_size=5,
            condition_on_previous_text=False,  # stops one hallucination ("thanks for watching") from repeating
            no_speech_threshold=0.5,
            log_prob_threshold=-0.8,           # drop low-confidence guesses instead of printing nonsense
        )
        # `segments` is a lazy generator: the real work (and any GPU error) happens here
        return [Segment(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]

    def transcribe(self, wav: Path, language: str | None) -> list[Segment]:
        try:
            return self._run(wav, language)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if not any(k in msg for k in ("cublas", "cudnn", "cuda")):
                raise TranscriptionError(str(exc)) from exc
            # GPU was auto-selected but CUDA libraries are missing: fall back to CPU.
            print("    (GPU not usable, falling back to CPU)", flush=True)
            self._model = self._WhisperModel(self._name, device="cpu", compute_type="int8")
            return self._run(wav, language)
