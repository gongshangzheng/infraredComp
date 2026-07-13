"""One-shot Xiph CIF natural-video dataset-level baseline.

Downloads the 6 CIF sequences (idempotent) if missing, runs stage 1 + stage 2
for each, and accumulates ALL runs into a SINGLE-DATASET results file
(results/video/xiph_cif.json) with a reproducibility envelope. Each run carries
dataset="Xiph-CIF-natural". Does NOT touch the default results.json — multiple
datasets coexist as separate files under results/video/.

Usage:
    uv run python scripts/run_natural_baseline.py
    uv run python scripts/run_natural_baseline.py --frames 30 --codecs x264,x265,vp9
    uv run python scripts/run_natural_baseline.py --skip-download
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

DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(PROJECT_ROOT / "datasets")))
XIPH_DIR = DATASETS_DIR / "raw" / "xiph_cif"
DATASET_NAME = "Xiph-CIF-natural"
# 独立数据集文件:不覆盖默认 results.json(多数据集共存)。
RESULTS_FILE = config.RESULTS_DIR / "xiph_cif.json"


def ensure_xiph() -> bool:
    """Ensure CIF y4m exist; invoke the downloader if missing."""
    if XIPH_DIR.is_dir() and any(XIPH_DIR.glob("*.y4m")):
        return True
    print("[xiph] no y4m found; downloading via download_xiph_natural.py ...")
    script = PROJECT_ROOT / "scripts" / "download_xiph_natural.py"
    proc = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print(f"[xiph] downloader failed (exit {proc.returncode})", file=sys.stderr)
        return False
    if not (XIPH_DIR.is_dir() and any(XIPH_DIR.glob("*.y4m"))):
        print("[xiph] still no y4m after download", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="One-shot Xiph CIF natural-video baseline.")
    ap.add_argument("--method", default="canny", choices=list_extractors())
    ap.add_argument("--crfs", default="18,23,28,33")
    ap.add_argument("--codecs", default=",".join(list_codecs()))
    ap.add_argument("--frames", type=int, default=None,
                    help="cap frame count per sequence (useful for slow AV1)")
    ap.add_argument("--skip-download", action="store_true",
                    help="don't invoke the downloader; skip missing y4m")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated seq stem subset (e.g. akiyo_cif,bus_cif); default=all")
    args = ap.parse_args()

    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]

    if not args.skip_download and not ensure_xiph():
        return 1

    config.ensure_dirs()

    # ----- Stage 1 per sequence (fault-isolated) -----
    seqs = sorted(XIPH_DIR.glob("*.y4m")) if XIPH_DIR.is_dir() else []
    if args.sequences:
        wanted = {s.strip() for s in args.sequences.split(",") if s.strip()}
        seqs = [s for s in seqs if s.stem in wanted]
    artifacts, used = [], []
    for y4m in seqs:
        try:
            print(f"[stage1] {y4m.name} (method={args.method})")
            art = extract_contour_video(str(y4m), method=args.method, frames=args.frames)
            print(f"[stage1] {y4m.name}: {art.frame_count} frames, "
                  f"{art.width}x{art.height}, {art.fps} fps")
            artifacts.append(art)
            used.append(str(y4m))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: {y4m.name} stage1 failed: {e} - skipping")

    if not artifacts:
        print("error: no sequences extracted", file=sys.stderr)
        return 1

    # ----- Stage 2: accumulate, save to the dataset-specific file -----
    print(f"[stage2] codecs={codecs} crfs={crfs} across {len(artifacts)} sequence(s)")
    all_results = []
    for art in artifacts:
        all_results.extend(
            run_benchmark(art, codecs=codecs, crfs=crfs, save=False, dataset=DATASET_NAME)
        )

    meta = build_metadata(
        inputs=used, codecs=codecs, crfs=crfs, method=args.method,
        frame_cap=args.frames, runner="scripts/run_natural_baseline.py",
        dataset=DATASET_NAME,
    )
    save_results_json(all_results, path=RESULTS_FILE, metadata=meta)
    print(f"[stage2] {len(all_results)} results -> {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
