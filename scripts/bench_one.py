"""Isolated single-(sequence, codec, *crfs) probe.

Runs one codec across all given CRFs for one sequence and emits one
``RESULT:<json>`` line per CRF to stdout.  Designed to be spawned one-per
(sequence, codec) so a native segfault in compressai's rans entropy coder
(accumulated across many runs in one process) only kills that one subprocess —
the parent sees a non-zero exit, keeps any RESULT lines already emitted, and
moves on.

Usage:
    python -u scripts/bench_one.py --sequence akiyo_cif --contour-dir \
        datasets/contour/akiyo_cif/canny --codec img-mbt2018 --crfs 1,4,8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.video import config
from benchmark.video.artifact_io import load_artifact
from benchmark.video.stage1_extract import extract_contour_video
from benchmark.video.stage2_benchmark import benchmark_codec
from benchmark.video.ffmpeg_util import run_ffmpeg


def _materialize_temp_frames(art):
    """If the artifact is video-based (PNGs deleted), decode the lossless
    contour video to a temp PNG dir (cropped to original dims) and return
    ``(temp_artifact, tmp_dir)``. Otherwise return ``(art, None)``.
    """
    if not art.video_path or art.frame_paths:
        return art, None
    tmp_dir = Path(tempfile.mkdtemp(prefix="cvbench_"))
    run_ffmpeg(["-y", "-i", art.video_path, "-vsync", "0",
                "-vf", f"crop={art.width}:{art.height}:0:0",
                "-pix_fmt", "gray", str(tmp_dir / "frame_%06d.png")])
    frame_paths = [str(p) for p in sorted(tmp_dir.glob("frame_*.png"))]
    return replace(art, frames_dir=str(tmp_dir), frame_paths=frame_paths), tmp_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--crfs", required=True, help="comma-separated CRF list")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--contour-dir", default=None,
                    help="reuse existing contour dir (skip stage1); else extract")
    ap.add_argument("--dataset", default="Xiph-CIF-natural")
    ap.add_argument("--checkpoint", default=None, help="trained checkpoint abs path（learned codec 覆盖权重）")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.contour_dir:
        art = load_artifact(args.contour_dir)
    else:
        y4m = config.raw_dir("xiph_cif") / f"{args.sequence}.y4m"
        if not y4m.is_file():
            print(json.dumps({"error": f"missing {y4m}"}))
            return 2
        art = extract_contour_video(str(y4m), method="canny", frames=args.frames)

    art, tmp_dir = _materialize_temp_frames(art)
    try:
        crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
        for crf in crfs:
            try:
                r = benchmark_codec(art, args.codec, crf, dataset=args.dataset, checkpoint_path=args.checkpoint)
                print("RESULT:" + json.dumps(r.to_dict()), flush=True)
            except Exception as e:  # noqa: BLE001
                msg = str(e).strip().splitlines()
                short = msg[0] if msg else str(e)
                print("ERROR:" + json.dumps({"id": f"{args.sequence}|{args.codec}|crf{crf}",
                                             "error": short[:200]}), flush=True)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
