"""CLI: python -m benchmark.video --input <raw_video|contour_dir> ...

Runs stage 1 (extract contour video) and/or stage 2 (compress + evaluate),
then writes results.json / charts / html report under results/video/.
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .codecs import list_codecs
from .extractors import list_extractors
from .stage1_extract import extract_contour_video
from .stage2_benchmark import run_benchmark
from .artifact_io import load_artifact
from .visualize import generate_report
from .html_report import generate_html_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Two-stage contour-video compression benchmark")
    parser.add_argument("--input", required=True,
                        help="raw video file/dir OR existing contour dir (with --skip-extract)")
    parser.add_argument("--method", default="canny", choices=list_extractors(),
                        help="edge extractor (stage 1)")
    parser.add_argument("--extract-only", action="store_true",
                        help="run only stage 1, produce datasets/contour/<name>/")
    parser.add_argument("--skip-extract", action="store_true",
                        help="skip stage 1; --input must point to an existing contour dir")
    parser.add_argument("--crfs", default="18,23,28,33",
                        help="comma-separated CRF list for stage 2")
    parser.add_argument("--codecs", default=",".join(list_codecs()),
                        help="comma-separated codec names")
    parser.add_argument("--frames", type=int, default=None,
                        help="cap frame count (useful for slow AV1)")
    parser.add_argument("--fps", type=float, default=None,
                        help="override fps (default: probed for video, 25 for frame dir)")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    crfs = [int(c) for c in args.crfs.split(",") if c.strip()]
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]

    # ----- Stage 1 -----
    if args.skip_extract:
        artifact = load_artifact(args.input)
        print(f"[stage1] skipped; loaded contour artifact: {artifact.source_name} "
              f"({artifact.frame_count} frames, {artifact.method})")
    else:
        print(f"[stage1] extracting contour video from {args.input} (method={args.method})")
        artifact = extract_contour_video(
            args.input, method=args.method, frames=args.frames, fps=args.fps)
        print(f"[stage1] done: {artifact.source_name} -> {artifact.frames_dir} "
              f"({artifact.frame_count} frames, {artifact.width}x{artifact.height}, "
              f"{artifact.fps} fps)")

    if args.extract_only:
        return 0

    # ----- Stage 2 -----
    print(f"[stage2] benchmarking codecs={codecs} crfs={crfs}")
    results = run_benchmark(artifact, codecs=codecs, crfs=crfs)
    print(f"[stage2] {len(results)} results -> {config.RESULTS_JSON}")

    if results:
        generate_report(results)
        generate_html_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
