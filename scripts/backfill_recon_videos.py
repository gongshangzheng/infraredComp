"""Backfill viewable mp4s for neural-codec runs from their existing recon PNGs.

Neural codecs (ssf2020 / img-* / dcvc_rt) write a non-playable ``.bin`` bitstream
plus decoded recon frames under ``results/video/recon/{tag}/``.  The speed/formal
pages can't play a ``.bin``, so before the recon→mp4 synth existed those runs
showed "无码流".  This script walks an existing results JSON and, for every
neural run whose recon dir is present but whose ``bitstreams/{tag}.mp4`` is
missing, synthesizes the mp4 from the PNGs — WITHOUT re-running the (slow)
codec inference.

Usage:
    python -u scripts/backfill_recon_videos.py
    python -u scripts/backfill_recon_videos.py --results results/video/xiph_cif.json --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.video import config
from benchmark.video.stage2_benchmark import synthesize_recon_video


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(config.RESULTS_DIR / "xiph_cif.json"))
    ap.add_argument("--force", action="store_true", help="regenerate even if mp4 exists")
    args = ap.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"results file not found: {results_path}", file=sys.stderr)
        return 1
    data = json.loads(results_path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])

    made, skipped, missing = 0, 0, 0
    for r in runs:
        # Only neural codecs need a synthesized recon video; traditional codecs
        # already produce a playable bitstream.
        if r.get("codec_family") != "learned-video":
            continue
        seq = r.get("sequence_name")
        codec = r.get("codec")
        crf = r.get("crf")
        if not seq or not codec or crf is None:
            continue
        tag = f"{seq}_{codec}_crf{crf}"
        recon_dir = config.RECON_DIR / tag
        mp4 = config.BITSTREAMS_DIR / f"{tag}.mp4"
        if not recon_dir.is_dir() or not any(recon_dir.glob("frame_*.png")):
            missing += 1
            continue
        if mp4.is_file() and not args.force:
            skipped += 1
            continue
        fps = r.get("fps") or 25.0
        try:
            synthesize_recon_video(recon_dir, fps, str(mp4))
            made += 1
            print(f"  made  {mp4.name}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {tag}: {e}")

    print(f"\nbackfill done: made={made} skipped={skipped} "
          f"missing-recon={missing} (of {len(runs)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
