"""Quick subset benchmark for a contour method (sobel/hed/...).

Runs stage1 (extract contour) + stage2 (benchmark grid) for the given sequences /
codecs / crfs and saves a per-method results file under results/video/. The file
name's stem controls the inferred mode (server `_load_results`: contains "_speed"
-> speed, else formal), so pass --mode speed for the 快速评测 page or omit for
正式评测.

Non-destructive: each (method, mode) gets its own file; existing canny data is
untouched. Only traditional (ffmpeg) codecs by default -> no learned-codec rans
segfault risk, safe in-process.

Usage:
    # formal subset (正式评测): akiyo+flower x traditional
    python scripts/bench_methods_subset.py
    # speed subset (快速评测): akiyo+bus x traditional, saved as *_speed.json
    python scripts/bench_methods_subset.py --mode speed --seqs akiyo_cif,bus_cif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.video.stage1_extract import extract_contour_video
from benchmark.video.stage2_benchmark import run_benchmark, save_results_json
from benchmark.video.repro import build_metadata
from benchmark.video import config
from benchmark.video.codecs import catalog as codec_catalog

DATASET = "Xiph-CIF-natural"
RAW_TPL = "datasets/raw/xiph_cif/{seq}.y4m"


def _crfs_for(codec: str, override: list[int] | None) -> list[int]:
    """override 非空 -> 用它; 否则取 catalog 里该 codec 的 qualities
    (mpeg4=[8,14,20,26], x264/x265/vp9=[18,23,28,33], learned 各自)。"""
    if override:
        return override
    for c in codec_catalog():
        if c["id"] == codec:
            return list(c["qualities"])
    return [18, 23, 28, 33]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="sobel,hed")
    ap.add_argument("--seqs", default="akiyo_cif,flower_cif")
    ap.add_argument("--codecs", default="x264,x265,vp9,mpeg4")
    ap.add_argument("--crfs", default="auto",
                    help="'auto' = 每 codec 取 catalog qualities; 否则逗号分隔整数, 所有 codec 共用")
    ap.add_argument("--mode", default="formal", choices=["formal", "speed"],
                    help="formal -> 正式评测; speed -> 快速评测 (file named *_speed.json)")
    ap.add_argument("--out-tag", default="",
                    help="输出文件名额外标签: xiph_cif_{method}{tag}{_speed?}.json")
    args = ap.parse_args()

    seqs = [s.strip() for s in args.seqs.split(",") if s.strip()]
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    override_crfs = None
    if args.crfs != "auto":
        override_crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    suffix = "_speed" if args.mode == "speed" else ""

    config.ensure_dirs()
    inputs = [RAW_TPL.format(seq=s) for s in seqs]
    for method in methods:
        all_results = []
        for seq in seqs:
            src = RAW_TPL.format(seq=seq)
            print(f"\n=== mode={args.mode} method={method} seq={seq} ===", flush=True)
            art = extract_contour_video(src, method=method)  # full frames; idempotent
            for codec in codecs:
                crfs = _crfs_for(codec, override_crfs)
                rs = run_benchmark(
                    art, codecs=[codec], crfs=crfs, save=False, dataset=DATASET,
                )
                all_results.extend(rs)
                print(f"  {seq}/{codec} crfs={crfs}: {len(rs)}", flush=True)
        meta = build_metadata(
            inputs=inputs, codecs=codecs, crfs=override_crfs or [],
            method=method, frame_cap=None,
            runner="scripts/bench_methods_subset.py", dataset=DATASET,
        )
        out = Path("results/video/xiph_cif_{m}{tag}{s}.json".format(
            m=method, tag=args.out_tag, s=suffix))
        save_results_json(all_results, path=out, metadata=meta)
        print(f"\n[{method}] {len(all_results)} runs -> {out}", flush=True)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
