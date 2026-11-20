"""Subtitle segments -> readable cues -> SRT / WebVTT text. Pure Python, no dependencies."""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass

END_PUNCT = (".", "!", "?", "…", ",", ";", ":")

# English words that should not be the last word of a cue (articles, prepositions, conjunctions...)
WEAK_END_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "with", "and", "or", "but", "that", "which",
    "as", "by", "from", "my", "your", "his", "her", "their", "our", "its", "if", "so", "than", "into",
}


@dataclass
class Segment:
    start: float  # seconds
    end: float
    text: str


# ----- timestamps ---------------------------------------------------------------------------

def format_timestamp(seconds: float, sep: str = ",") -> str:
    """90.5 -> '00:01:30,500' (SRT uses ',', WebVTT uses '.')."""
    ms_total = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms_total, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def parse_timestamp(text: str) -> float:
    m = _TS.fullmatch(text.strip())
    if not m:
        raise ValueError(f"bad timestamp: {text!r}")
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


# ----- splitting and wrapping ---------------------------------------------------------------

def split_segment(seg: Segment, max_chars: int = 42, max_lines: int = 2, max_duration: float = 6.0) -> list[Segment]:
    """Split a long segment into cues that fit on screen. Timing is shared out by character count."""
    text = " ".join(seg.text.split())
    if not text:
        return []
    duration = max(seg.end - seg.start, 0.0)
    limit = max_chars * max_lines
    n = max(1, math.ceil(len(text) / limit), math.ceil(duration / max_duration) if duration else 1)
    if n == 1:
        return [Segment(seg.start, seg.end, text)]

    target = len(text) / n
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_len = 0
    for word in text.split():
        add = len(word) + (1 if cur else 0)
        if cur and (
            cur_len + add > limit
            or cur_len + add > target * 1.2
            or (cur_len >= target * 0.6 and cur[-1].endswith(END_PUNCT))
        ):
            chunks.append(cur)
            cur, cur_len = [], 0
            add = len(word)
        cur.append(word)
        cur_len += add
    if cur:
        # don't leave a tiny orphan at the end if it fits into the previous cue
        if chunks and len(" ".join(cur)) < target * 0.25 and len(" ".join(chunks[-1] + cur)) <= limit:
            chunks[-1] += cur
        else:
            chunks.append(cur)

    # never end a cue on a word that needs its successor ("...that has a" | "Blue Yeti microphone")
    for i in range(len(chunks) - 1):
        while (
            len(chunks[i]) > 1
            and chunks[i][-1].lower().strip(".,;:!?…\"'") in WEAK_END_WORDS
            and not chunks[i][-1].endswith(END_PUNCT)
            and len(" ".join([chunks[i][-1]] + chunks[i + 1])) <= limit
        ):
            chunks[i + 1].insert(0, chunks[i].pop())

    pieces = [" ".join(c) for c in chunks]
    total_chars = sum(len(p) for p in pieces)
    out: list[Segment] = []
    t = seg.start
    for i, piece in enumerate(pieces):
        end = seg.end if i == len(pieces) - 1 else t + duration * len(piece) / total_chars
        out.append(Segment(t, end, piece))
        t = end
    return out


def wrap_lines(text: str, max_chars: int = 42, max_lines: int = 2) -> str:
    """Break text into balanced lines (a 2-line cue gets two similar-length halves, not 42 + 3)."""
    text = " ".join(text.split())
    if len(text) <= max_chars or max_lines <= 1:
        return text
    words = text.split()
    if max_lines == 2:
        best, best_score = None, None
        for i in range(1, len(words)):
            a, b = " ".join(words[:i]), " ".join(words[i:])
            if len(a) > max_chars or len(b) > max_chars:
                continue
            score = abs(len(a) - len(b)) - (4 if a.endswith(END_PUNCT) else 0)
            if best_score is None or score < best_score:
                best, best_score = (a, b), score
        if best:
            return "\n".join(best)
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines:  # should not happen after split_segment; keep all text anyway
        lines = lines[: max_lines - 1] + [" ".join(lines[max_lines - 1:])]
    return "\n".join(lines)


def _fix_dangling_words(segments: list[Segment], max_gap: float = 1.5) -> list[Segment]:
    """Whisper sometimes ends a segment on 'a' / 'the' / 'with'. Move such trailing words to the start of the
    next segment (timing shared by character count) so no cue ends in the middle of a phrase."""
    segs = [Segment(s.start, s.end, " ".join(s.text.split())) for s in segments]
    for i in range(len(segs) - 1):
        seg, nxt = segs[i], segs[i + 1]
        words = seg.text.split()
        moved: list[str] = []
        while (
            len(words) > 1
            and words[-1].lower().strip(".,;:!?…\"'") in WEAK_END_WORDS
            and not words[-1].endswith(END_PUNCT)
        ):
            moved.insert(0, words.pop())
        if not moved or nxt.start - seg.end > max_gap or not nxt.text:
            continue
        share = len(" ".join(moved)) / max(len(seg.text), 1)
        boundary = seg.end - (seg.end - seg.start) * share
        seg.text, seg.end = " ".join(words), boundary
        nxt.text = " ".join(moved) + " " + nxt.text
        nxt.start = min(nxt.start, boundary)
    return segs


def refine(
    segments: list[Segment],
    max_chars: int = 42,
    max_lines: int = 2,
    max_duration: float = 6.0,
    min_duration: float = 0.7,
) -> list[Segment]:
    """Raw recognizer output -> clean cues: no empties, split to fit, wrapped, no overlaps."""
    cues: list[Segment] = []
    for seg in _fix_dangling_words(sorted(segments, key=lambda s: s.start)):
        for piece in split_segment(seg, max_chars, max_lines, max_duration):
            cues.append(Segment(piece.start, piece.end, wrap_lines(piece.text, max_chars, max_lines)))

    for i, cue in enumerate(cues):
        next_start = cues[i + 1].start if i + 1 < len(cues) else math.inf
        if cue.end - cue.start < min_duration:           # too short to read: extend if there is room
            cue.end = min(cue.start + min_duration, next_start)
        if cue.end > next_start:                          # overlap: end where the next cue starts
            cue.end = next_start
        if cue.end <= cue.start:
            cue.end = cue.start + 0.001
    return [c for c in cues if c.text]


# ----- output -------------------------------------------------------------------------------

def to_srt(cues: list[Segment]) -> str:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}\n")
    return "\n".join(blocks)


def to_vtt(cues: list[Segment]) -> str:
    blocks = ["WEBVTT\n"]
    for c in cues:
        blocks.append(f"{format_timestamp(c.start, '.')} --> {format_timestamp(c.end, '.')}\n{c.text}\n")
    return "\n".join(blocks)


def parse_srt(text: str) -> list[Segment]:
    """Read SRT back in (used by the tests and handy for validation)."""
    out: list[Segment] = []
    for block in re.split(r"\n\s*\n", text.strip().replace("\r\n", "\n")):
        lines = block.split("\n")
        if len(lines) < 3 or "-->" not in lines[1]:
            raise ValueError(f"bad SRT block: {block!r}")
        a, b = (x.strip() for x in lines[1].split("-->"))
        out.append(Segment(parse_timestamp(a), parse_timestamp(b), "\n".join(lines[2:])))
    return out
