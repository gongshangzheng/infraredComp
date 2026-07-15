"""End-to-end self-check: synthesize a tiny video, run stage 1 + stage 2,
assert sane metrics + the odd-dimension pad path. Doubles as the first run.

Usage: uv run python -m benchmark.video.verify
"""

from __future__ import annotations

import math
import tempfile
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config
from .ffmpeg_util import find_ffmpeg
from .stage1_extract import extract_contour_video, load_contour_video_frames
from .stage2_benchmark import run_benchmark
from .visualize import generate_report
from .html_report import generate_html_report


def _make_frames(frames_dir: Path, n: int, h: int, w: int) -> None:
    """Synthesize moving-circle grayscale frames (even or odd dims)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = np.zeros((h, w), np.uint8)
        cx = 10 + i * max(1, (w - 20) // max(1, n - 1))
        cv2.circle(img, (cx, h // 2), 8, 255, -1)
        cv2.rectangle(img, (0, h - 12), (w - 1, h - 1), 200, -1)
        cv2.imwrite(str(frames_dir / f"f{i:02d}.png"), img)


def _make_video(frames_dir: Path, n: int = 8, h: int = 64, w: int = 64) -> Path:
    """Synthesize a moving-circle grayscale video (even dims for yuv420p mux)."""
    _make_frames(frames_dir, n, h, w)
    mp4 = frames_dir.parent / "verify.mp4"
    subprocess.run(
        [find_ffmpeg(), "-y", "-framerate", "10", "-i", str(frames_dir / "f%02d.png"),
         "-pix_fmt", "yuv420p", "-vf", "format=gray", str(mp4)],
        capture_output=True, check=True,
    )
    return mp4


def _check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cvverify_"))
    print(f"verify temp: {tmp}")
    all_ok = True

    # ---- Even dims (64x64) ----
    even_raw = tmp / "even_raw"
    mp4 = _make_video(even_raw, n=8, h=64, w=64)
    art = extract_contour_video(mp4, method="canny", out_dir=tmp / "even_contour")
    all_ok &= _check("even: manifest exists", Path(art.manifest_path).exists())
    all_ok &= _check("even: frame_count == 8", art.frame_count == 8)
    all_ok &= _check("even: dims 64x64", (art.height, art.width) == (64, 64))
    gt = load_contour_video_frames(art)
    all_ok &= _check("even: gt shape (8,64,64) uint8",
                     gt.shape == (8, 64, 64) and gt.dtype == np.uint8)

    res = run_benchmark(art, codecs=["x264", "vp9"], crfs=[23])
    all_ok &= _check("even: 2 results", len(res) == 2)
    for r in res:
        ok = (math.isfinite(r.psnr) and math.isfinite(r.ssim)
              and r.compressed_bytes > 0 and r.enc_fps > 0 and r.dec_fps > 0
              and r.bitrate_kbps > 0)
        all_ok &= _check(f"even: {r.codec} crf23 sane (psnr={r.psnr:.2f} ssim={r.ssim:.4f} "
                         f"bpp={r.bpp:.3f} enc={r.enc_fps:.0f}fps dec={r.dec_fps:.0f}fps)", ok)

    # ---- Odd dims (65x63) -> frame-dir input + pad path ----
    odd_raw = tmp / "odd_raw"
    _make_frames(odd_raw, n=6, h=63, w=65)  # odd dims can't be yuv420p-muxed; use frame dir
    arto = extract_contour_video(odd_raw, method="canny", out_dir=tmp / "odd_contour")
    all_ok &= _check("odd: dims 63x65", (arto.height, arto.width) == (63, 65))
    rodd = run_benchmark(arto, codecs=["x264"], crfs=[23])
    if rodd:
        r = rodd[0]
        all_ok &= _check(f"odd: x264 pad path works (psnr={r.psnr:.2f} bytes={r.compressed_bytes})",
                         math.isfinite(r.psnr) and r.compressed_bytes > 0)
    else:
        all_ok &= _check("odd: x264 produced a result", False)

    # ---- Reporting ----
    rep = generate_report(res)
    all_ok &= _check("report: summary has rows", len(rep["summary"]) >= 1)
    hp = generate_html_report(res)
    all_ok &= _check("report.html exists", Path(hp).exists())
    all_ok &= _check("results.json exists", Path(config.RESULTS_JSON).exists())

    print("\n" + ("=== ALL PASS ===" if all_ok else "=== SOME FAIL ==="))
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
