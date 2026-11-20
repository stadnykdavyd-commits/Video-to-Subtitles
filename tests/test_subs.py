"""Run:  python -m unittest discover -s tests -v   (ffmpeg needed only for the end-to-end tests)"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from subs.ffmpeg_utils import FFmpegError, probe_duration  # noqa: E402
from subs.pipeline import process_file  # noqa: E402
from subs.srt import (  # noqa: E402
    Segment, format_timestamp, parse_srt, parse_timestamp, refine, split_segment, to_srt, to_vtt, wrap_lines,
)

LONG = ("This is a fairly long sentence that cannot possibly fit on a single subtitle line. "
        "Here comes a second sentence, which makes the whole segment much longer than the screen allows.")


class TimestampTests(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_timestamp(0), "00:00:00,000")
        self.assertEqual(format_timestamp(90.5), "00:01:30,500")
        self.assertEqual(format_timestamp(3661.0015), "01:01:01,002")
        self.assertEqual(format_timestamp(1.5, "."), "00:00:01.500")

    def test_roundtrip(self):
        for s in (0.0, 1.234, 59.999, 3600.5, 7322.001):
            self.assertAlmostEqual(parse_timestamp(format_timestamp(s)), s, places=3)

    def test_negative_is_clamped(self):
        self.assertEqual(format_timestamp(-1), "00:00:00,000")


class WrapAndSplitTests(unittest.TestCase):
    def test_short_text_untouched(self):
        self.assertEqual(wrap_lines("Hello world"), "Hello world")

    def test_two_lines_are_balanced_and_fit(self):
        out = wrap_lines("This subtitle is a little too long for one single line")
        a, b = out.split("\n")
        self.assertLessEqual(len(a), 42)
        self.assertLessEqual(len(b), 42)
        self.assertLess(abs(len(a) - len(b)), 15)

    def test_long_segment_is_split_with_all_text_kept(self):
        parts = split_segment(Segment(0, 12, LONG))
        self.assertGreater(len(parts), 1)
        self.assertEqual(" ".join(p.text for p in parts), LONG)
        self.assertTrue(all(len(p.text) <= 84 for p in parts))
        self.assertAlmostEqual(parts[0].start, 0)
        self.assertAlmostEqual(parts[-1].end, 12)
        for a, b in zip(parts, parts[1:]):
            self.assertAlmostEqual(a.end, b.start)

    def test_long_duration_is_split(self):
        parts = split_segment(Segment(0, 20, "word " * 12))
        self.assertGreaterEqual(len(parts), 4)

    def test_empty_text_dropped(self):
        self.assertEqual(refine([Segment(0, 1, "   "), Segment(1, 2, "")]), [])


class RefineTests(unittest.TestCase):
    def test_overlap_is_fixed_and_order_restored(self):
        cues = refine([Segment(5, 9, "second"), Segment(0, 6, "first")])
        self.assertEqual([c.text for c in cues], ["first", "second"])
        self.assertLessEqual(cues[0].end, cues[1].start)

    def test_short_cue_extended_when_there_is_room(self):
        cues = refine([Segment(0, 0.2, "Hi"), Segment(5, 6, "there")])
        self.assertAlmostEqual(cues[0].end - cues[0].start, 0.7)

    def test_short_cue_not_extended_into_next(self):
        cues = refine([Segment(0, 0.2, "Hi"), Segment(0.4, 2, "there")])
        self.assertLessEqual(cues[0].end, 0.4)


class OutputTests(unittest.TestCase):
    def test_srt_layout_and_roundtrip(self):
        cues = [Segment(1, 2.5, "Hello"), Segment(3, 4, "two\nlines")]
        text = to_srt(cues)
        self.assertTrue(text.startswith("1\n00:00:01,000 --> 00:00:02,500\nHello\n"))
        back = parse_srt(text)
        self.assertEqual([c.text for c in back], ["Hello", "two\nlines"])
        self.assertAlmostEqual(back[1].start, 3)

    def test_vtt_header_and_dots(self):
        text = to_vtt([Segment(1, 2.5, "Hello")])
        self.assertTrue(text.startswith("WEBVTT"))
        self.assertIn("00:00:01.000 --> 00:00:02.500", text)


class FakeBackend:
    def __init__(self):
        self.seen_wav_info = None

    def transcribe(self, wav, language):
        self.seen_wav_info = (wav.name, wav.stat().st_size, language)
        return [Segment(0.0, 1.0, "First line."), Segment(1.0, 2.5, LONG)]


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not installed")
class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.video = self.tmp / "clip.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=160x120:d=3",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=3",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(self.video),
        ], check=True)
        self.original = self.video.read_bytes()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pipeline_writes_srt_and_vtt_and_keeps_original(self):
        backend = FakeBackend()
        res = process_file(self.video, self.tmp / "out", backend, language="en", formats=("srt", "vtt"))
        self.assertEqual({p.suffix for p in res.outputs}, {".srt", ".vtt"})
        cues = parse_srt((self.tmp / "out" / "clip.srt").read_text(encoding="utf-8"))
        self.assertEqual(len(cues), res.cues)
        self.assertGreater(res.cues, 2)  # the long segment was split
        self.assertTrue(all(max(len(l) for l in c.text.split("\n")) <= 42 for c in cues))
        self.assertEqual(self.video.read_bytes(), self.original)   # original untouched
        self.assertEqual(backend.seen_wav_info[0], "audio.wav")
        self.assertEqual(backend.seen_wav_info[2], "en")

    def test_audio_is_16k_mono_wav(self):
        seen = {}

        class Spy:
            def transcribe(self, wav, language):
                out = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels,codec_name",
                     "-of", "csv=p=0", str(wav)], capture_output=True, text=True).stdout.strip()
                seen["info"] = out
                return []

        process_file(self.video, self.tmp / "out2", Spy())
        self.assertEqual(seen["info"], "pcm_s16le,16000,1")

    def test_no_audio_stream_gives_clear_error(self):
        silent = self.tmp / "silent.mp4"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=red:s=160x120:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent),
        ], check=True)
        with self.assertRaises(FFmpegError):
            process_file(silent, self.tmp / "out3", FakeBackend())

    def test_probe_duration(self):
        self.assertAlmostEqual(probe_duration(self.video), 3.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
