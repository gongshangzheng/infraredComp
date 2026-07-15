"""One-off: transcode MPEG-4 Part 2 bitstreams to browser-playable H.264.

Browsers can't decode MPEG-4 Part 2 video, so <video> can't show the raw mpeg4
bitstreams. benchmark_codec now synthesizes a viewable H.264 mp4 for new mpeg4
runs (browser_playable=False), but old Part 2 bitstreams on disk still aren't
playable. This script finds every `*_mpeg4_crf*.mp4` whose video codec isn't
already h264 and transcodes it in place (Part 2 -> H.264 crf18, near-lossless).
Result files / metrics untouched (compressed_bytes already recorded).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BITSTREAMS = PROJECT_ROOT / "results" / "video" / "bitstreams"


def _ffmpeg() -> str:
    sys.path.insert(0, str(PROJECT_ROOT))
    from benchmark.video.ffmpeg_util import find_ffmpeg, find_ffprobe  # noqa: E402
    return find_ffmpeg(), find_ffprobe()


def _codec(ffprobe: str, path: Path) -> str:
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def main() -> int:
    ffmpeg, ffprobe = _ffmpeg()
    targets = sorted(BITSTREAMS.glob("*_mpeg4_crf*.mp4"))
    fixed = skipped = failed = 0
    for p in targets:
        if _codec(ffprobe, p) == "h264":
            skipped += 1
            continue
        tmp = p.with_suffix(".view.mp4")
        r = subprocess.run(
            [ffmpeg, "-y", "-i", str(p), "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", str(tmp)],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not tmp.is_file():
            failed += 1
            print(f"[fail] {p.name}: {r.stderr[-200:]}")
            tmp.unlink(missing_ok=True)
            continue
        tmp.replace(p)
        fixed += 1
        print(f"[fixed] {p.name}")
    print(f"\nfixed={fixed} skipped(h264)={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
