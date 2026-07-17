#!/usr/bin/env python3
"""BSDS val 轮廓 GT 的 codec 评测 runner。

把 BSDS500 val 的 ground-truth 软边缘图（datasets/contour/bsds_val_gt/frame_*.png）
每张视为 1 帧的伪视频序列，跑 stage-2 视频 codec 评测。

formal  → results/video/bsds_val.json（全部图片）
speed   → results/video/bsds_val_speed.json（默认前 50 张，可 --max-images/--sequences）

用法：
    python scripts/run_bsds_baseline.py --mode formal --codecs x264,x265 --crfs 18,23,28,33
    python scripts/run_bsds_baseline.py --mode speed --max-images 20
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.video import config  # type: ignore
from benchmark.video.codecs import catalog  # type: ignore
from benchmark.video.data import ContourArtifact  # type: ignore
from benchmark.video.repro import build_metadata  # type: ignore
from benchmark.video.stage2_benchmark import run_benchmark, save_results_json  # type: ignore

DATASET_NAME = "BSDS-val"
RESULTS_FILE_BY_MODE = {
    "formal": config.RESULTS_DIR / "bsds_val.json",
    "speed": config.RESULTS_DIR / "bsds_val_speed.json",
}


def load_bsds_val_artifacts(max_images: int | None = None, sequences: str | None = None) -> list[ContourArtifact]:
    """从 datasets/contour/bsds_val_gt/manifest.json 加载每张图为 1 帧 ContourArtifact。"""
    gt_dir = config.DATASETS_DIR / "contour" / "bsds_val_gt"
    manifest_path = gt_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"BSDS val GT 未找到：{manifest_path}。"
            f"请先运行：python scripts/convert_bsds_gt.py --splits val"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_dir = Path(manifest.get("frames_dir") or str(gt_dir))
    pngs = sorted(frames_dir.glob("frame_*.png"))
    if sequences:
        want = {s.strip() for s in sequences.split(",") if s.strip()}
        pngs = [p for p in pngs if p.stem in want]
    if max_images is not None:
        pngs = pngs[:max_images]
    if not pngs:
        raise FileNotFoundError(f"BSDS val GT 目录下无可用图片：{frames_dir}")

    artifacts: list[ContourArtifact] = []
    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[warn] 无法读取 {p}，跳过", flush=True)
            continue
        h, w = img.shape
        artifacts.append(ContourArtifact(
            source_name=p.stem,
            method="gt",
            frames_dir=str(frames_dir),
            frame_paths=[str(p)],
            frame_count=1,
            fps=25.0,
            width=w,
            height=h,
            duration_s=1 / 25.0,
            manifest_path=str(manifest_path),
            video_path="",
        ))
    return artifacts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="formal", choices=["formal", "speed"],
                    help="formal→全部图片；speed→默认前 50 张")
    ap.add_argument("--codecs", default=None, help="逗号分隔 codec id；空=全部")
    ap.add_argument("--crfs", default=None, help="逗号分隔 CRF；空=18,23,28,33（仅传统 codec）")
    ap.add_argument("--sequences", default=None, help="逗号分隔图片 stem（speed 子集）")
    ap.add_argument("--max-images", type=int, default=None,
                    help="最多跑前 N 张图；覆盖 speed 默认 50")
    ap.add_argument("--checkpoint", default=None, help="学习式 codec 权重覆盖")
    args = ap.parse_args()

    config.ensure_dirs()

    max_images = args.max_images
    if args.mode == "speed" and max_images is None and not args.sequences:
        max_images = 50

    artifacts = load_bsds_val_artifacts(max_images=max_images, sequences=args.sequences)
    print(f"[bsds] loaded {len(artifacts)} images from bsds_val_gt", flush=True)

    codec_ids = [c["id"] for c in catalog()]
    if args.codecs:
        want = {c.strip() for c in args.codecs.split(",") if c.strip()}
        codec_ids = [c for c in codec_ids if c in want]
    if not codec_ids:
        raise SystemExit("无可用 codec，请检查 --codecs")

    # 每个 codec 的 quality 参数：传统 codec 默认 18/23/28/33，学习式用 catalog。
    _LEARNED_IDS = {c["id"] for c in catalog() if c["kind"] != "codec"}
    CODEC_CRF_MAP: dict[str, list[int]] = {cid: list(next(c for c in catalog() if c["id"] == cid)["qualities"]) for cid in codec_ids}
    if args.crfs:
        override = [int(c) for c in args.crfs.split(",") if c.strip()]
        for cid in codec_ids:
            if cid not in _LEARNED_IDS:
                CODEC_CRF_MAP[cid] = override

    all_results = []
    for art in artifacts:
        print(f"[bsds] {art.source_name} {art.width}x{art.height}", flush=True)
        # 视频 codec 期望顺序命名帧目录；把单张图复制到临时 dir/frame_000001.png。
        tmp_dir = Path(tempfile.mkdtemp(prefix="bsds_"))
        tmp_png = tmp_dir / "frame_000001.png"
        shutil.copy2(art.frame_paths[0], tmp_png)
        work = ContourArtifact(
            source_name=art.source_name,
            method=art.method,
            frames_dir=str(tmp_dir),
            frame_paths=[str(tmp_png)],
            frame_count=1,
            fps=art.fps,
            width=art.width,
            height=art.height,
            duration_s=art.duration_s,
            manifest_path=art.manifest_path,
            video_path="",
        )
        try:
            for cid in codec_ids:
                results = run_benchmark(
                    work,
                    codecs=[cid],
                    crfs=CODEC_CRF_MAP[cid],
                    save=False,
                    dataset=DATASET_NAME,
                    checkpoint_path=args.checkpoint,
                )
                for r in results:
                    if args.checkpoint:
                        r.id = f"{r.id}|ckpt={os.path.basename(args.checkpoint)}"
                all_results.extend(results)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    all_crfs = sorted({c for crfs in CODEC_CRF_MAP.values() for c in crfs})
    metadata = build_metadata(
        inputs=[a.frame_paths[0] for a in artifacts],
        codecs=codec_ids,
        crfs=all_crfs,
        method="gt",
        frame_cap=None,
        runner="scripts/run_bsds_baseline.py",
        dataset=DATASET_NAME,
    )
    metadata["mode"] = args.mode
    if args.checkpoint:
        metadata["checkpoint"] = args.checkpoint

    out = RESULTS_FILE_BY_MODE[args.mode]
    save_results_json(all_results, path=out, metadata=metadata)
    print(f"[bsds] done: {len(all_results)} results -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
