"""One-shot OSU Color-Thermal dataset-level baseline.

Downloads OSU (idempotent) if any sequence is missing, runs stage 1 + stage 2
for seq1..6, accumulates ALL runs into a single multi-sequence results.json
with a reproducibility envelope, and writes the charts + HTML report.

Usage:
    uv run python scripts/run_osu_baseline.py
    uv run python scripts/run_osu_baseline.py --frames 30 --codecs x264,x265,vp9
    uv run python scripts/run_osu_baseline.py --skip-download   # data must exist

Sequences live under ${INFRACOMP_DATASETS_DIR}/raw/osu_color_thermal/seqN.mp4.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.video import config
from benchmark.video.codecs import list_codecs
from benchmark.video.extractors import list_extractors
from benchmark.video.repro import build_metadata
from benchmark.video.stage1_extract import extract_contour_video
from benchmark.video.stage2_benchmark import run_benchmark, save_results_json
from benchmark.video.visualize import generate_report
from benchmark.video.html_report import generate_html_report

DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(PROJECT_ROOT / "datasets")))
OSU_DIR = config.raw_dir("osu_color_thermal")
DATASET_NAME = "OSU Color-Thermal (OTCBVS Dataset 03)"
SEQ_RANGE = range(1, 7)  # seq1..6


def ensure_osu() -> bool:
    """Ensure all seqN.mp4 exist; invoke the downloader if any are missing."""
    missing = [i for i in SEQ_RANGE if not (OSU_DIR / f"seq{i}.mp4").exists()]
    if not missing:
        return True
    print(f"[osu] missing seq{missing}; downloading via download_osu_color_thermal.py ...")
    script = PROJECT_ROOT / "scripts" / "download_osu_color_thermal.py"
    proc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"[osu] downloader failed (exit {proc.returncode})", file=sys.stderr)
        return False
    still_missing = [i for i in SEQ_RANGE if not (OSU_DIR / f"seq{i}.mp4").exists()]
    if still_missing:
        print(f"[osu] seq{still_missing} still missing after download", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="One-shot OSU Color-Thermal baseline.")
    ap.add_argument("--method", default="canny", choices=list_extractors())
    ap.add_argument("--crfs", default="18,23,28,33")
    ap.add_argument("--codecs", default=",".join(list_codecs()))
    ap.add_argument("--frames", type=int, default=None,
                    help="cap frame count per sequence (useful for slow AV1)")
    ap.add_argument("--skip-download", action="store_true",
                    help="don't invoke the downloader; skip missing seqN.mp4")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated seq stem subset (e.g. seq1,seq3); default=all")
    args = ap.parse_args()

    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]

    if not args.skip_download and not ensure_osu():
        return 1

    config.ensure_dirs()

    # ----- Stage 1 per sequence (fault-isolated) -----
    seq_indices = list(SEQ_RANGE)
    if args.sequences:
        wanted = {s.strip() for s in args.sequences.split(",") if s.strip()}
        seq_indices = [i for i in seq_indices if f"seq{i}" in wanted]
    artifacts, used = [], []
    for idx in seq_indices:
        mp4 = OSU_DIR / f"seq{idx}.mp4"
        if not mp4.exists():
            print(f"  WARN: seq{idx}.mp4 missing, skipping")
            continue
        try:
            print(f"[stage1] seq{idx} <- {mp4} (method={args.method})")
            art = extract_contour_video(str(mp4), method=args.method, frames=args.frames)
            print(f"[stage1] seq{idx}: {art.frame_count} frames, "
                  f"{art.width}x{art.height}, {art.fps} fps")
            artifacts.append(art)
            used.append(str(mp4))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: seq{idx} stage1 failed: {e} — skipping")

    if not artifacts:
        print("error: no sequences extracted", file=sys.stderr)
        return 1

    # ----- Stage 2: accumulate across sequences, save once -----
    print(f"[stage2] codecs={codecs} crfs={crfs} across {len(artifacts)} sequence(s)")
    all_results = []
    for art in artifacts:
        all_results.extend(run_benchmark(art, codecs=codecs, crfs=crfs, save=False))

    meta = build_metadata(
        inputs=used, codecs=codecs, crfs=crfs, method=args.method,
        frame_cap=args.frames, runner="scripts/run_osu_baseline.py",
        dataset=DATASET_NAME,
    )
    save_results_json(all_results, metadata=meta)
    print(f"[stage2] {len(all_results)} results -> {config.RESULTS_JSON}")

    if all_results:
        generate_report(all_results)
        generate_html_report(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
