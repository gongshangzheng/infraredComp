"""CLI: python -m benchmark.video --input <raw_video|dir|contour_dir> ...

Runs stage 1 (extract contour video) and/or stage 2 (compress + evaluate),
then writes results.json / charts / html report under results/video/.

--input is repeatable; a directory is globbed for video files (extract mode)
or treated as one contour dir (--skip-extract). All sequences accumulate into
a single multi-sequence results.json (a dataset-level baseline).
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .codecs import list_codecs
from .extractors import list_extractors
from .repro import build_metadata
from .stage1_extract import extract_contour_video, expand_inputs
from .stage2_benchmark import run_benchmark, save_results_json
from .artifact_io import load_artifact
from .visualize import generate_report
from .html_report import generate_html_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Two-stage contour-video compression benchmark")
    parser.add_argument("--input", required=True, action="append", metavar="PATH",
                        help="raw video file/dir OR contour dir; repeatable; "
                             "dirs are globbed for videos in extract mode")
    parser.add_argument("--method", default="canny", choices=list_extractors(),
                        help="edge extractor (stage 1)")
    parser.add_argument("--extract-only", action="store_true",
                        help="run only stage 1, produce datasets/contour/<name>/")
    parser.add_argument("--skip-extract", action="store_true",
                        help="skip stage 1; each --input must be an existing contour dir")
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

    # ----- Stage 1: expand inputs, extract/load each artifact -----
    inputs = expand_inputs(args.input, skip_extract=args.skip_extract)
    if not inputs:
        print("error: no input paths resolved from --input", file=sys.stderr)
        return 2

    artifacts = []
    for src in inputs:
        if args.skip_extract:
            artifact = load_artifact(src)
            print(f"[stage1] skipped; loaded contour artifact: {artifact.source_name} "
                  f"({artifact.frame_count} frames, {artifact.method})")
        else:
            print(f"[stage1] extracting contour video from {src} (method={args.method})")
            artifact = extract_contour_video(
                src, method=args.method, frames=args.frames, fps=args.fps)
            print(f"[stage1] done: {artifact.source_name} -> {artifact.frames_dir} "
                  f"({artifact.frame_count} frames, {artifact.width}x{artifact.height}, "
                  f"{artifact.fps} fps)")
        artifacts.append(artifact)

    if args.extract_only:
        return 0

    # ----- Stage 2: run grid per sequence, accumulate, save once -----
    print(f"[stage2] benchmarking codecs={codecs} crfs={crfs} "
          f"across {len(artifacts)} sequence(s)")
    all_results = []
    for artifact in artifacts:
        results = run_benchmark(artifact, codecs=codecs, crfs=crfs, save=False)
        all_results.extend(results)

    meta = build_metadata(
        inputs=inputs, codecs=codecs, crfs=crfs, method=args.method,
        frame_cap=args.frames, runner="python -m benchmark.video",
    )
    save_results_json(all_results, metadata=meta)
    print(f"[stage2] {len(all_results)} results -> {config.RESULTS_JSON}")

    if all_results:
        generate_report(all_results)
        generate_html_report(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
