#!/usr/bin/env python3
"""一次性预提取 imagenet 边缘到 PNG（落地），供训练快速读取（GPU 喂得满）。

遍历指定 split 的全部 parquet 行（默认全量），用给定方法（hed/canny/sobel）
提取边缘，resize 到 `--size` 存为 frame_XXXXXXX.png + manifest.json。
多进程分片加速；skip-if-exists 可断点续跑。

产物目录：datasets/contour/imagenet_<split>_<method>/frame_XXXXXXX.png
训练侧 resolve_training_dataset 检测到该目录有 manifest 即改用 ContourPNGDataset
（读 PNG，快），否则回退流式在线提取。

用法：
  python scripts/extract_imagenet_contour.py --split train --method hed --workers 8
  python scripts/extract_imagenet_contour.py --split train --method hed --limit 64   # 小批量验证
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(REPO / "datasets")))
_SPLIT_PREFIX = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def build_index(split: str, shards: int) -> tuple[list[str], list[tuple[int, int, int]]]:
    """返回 (file_paths, index)，index = 全部行的 (file_idx, row_group, row_in_group)。"""
    import pyarrow.parquet as pq
    data_dir = DATASETS_DIR / "imagenet" / "data"
    prefix = _SPLIT_PREFIX.get(split, split)
    files = sorted(data_dir.glob(f"{prefix}-*.parquet"))
    if not files:
        raise RuntimeError(f"无 imagenet parquet: {data_dir}/{prefix}-*.parquet")
    if shards and shards > 0:
        files = files[: max(1, int(shards))]
    file_paths = [str(f) for f in files]
    index: list[tuple[int, int, int]] = []
    for fi, fp in enumerate(file_paths):
        md = pq.ParquetFile(fp).metadata
        for rg in range(md.num_row_groups):
            n = md.row_group(rg).num_rows
            for r in range(n):
                index.append((fi, rg, r))
    return file_paths, index


# ---- worker 进程状态（spawn 下每进程独立建 extractor / ParquetFile）----------- #
_W: dict = {}


def _init(file_paths: list[str], method: str, size: int) -> None:
    import pyarrow.parquet as pq
    from benchmark.video.extractors import build_extractor
    _W["pq"] = [pq.ParquetFile(fp) for fp in file_paths]
    _W["ex"] = build_extractor(method)
    _W["size"] = size


def _one(task) -> int:
    i, fi, rg, r, out_dir, size = task
    out = Path(out_dir) / f"frame_{i:07d}.png"
    if out.exists():
        return 0  # skip-if-exists（续跑）
    col = _W["pq"][fi].read_row_group(rg, columns=["image"]).column("image")
    cell = col[r].as_py()
    b = cell["bytes"] if isinstance(cell, dict) else cell
    img = Image.open(io.BytesIO(b)).convert("L")
    arr = np.array(img, dtype=np.uint8)
    edges = _W["ex"].extract(arr)
    if edges.dtype != np.uint8:
        edges = edges.astype(np.uint8)
    eimg = Image.fromarray(edges)
    if eimg.size != (size, size):  # PIL size=(W,H)
        eimg = eimg.resize((size, size), Image.BILINEAR)
    eimg.save(out)
    return 1


def _tasks(file_paths, index, out_dir, size):
    for i, (fi, rg, r) in enumerate(index):
        yield (i, fi, rg, r, out_dir, size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--method", default="hed", help="轮廓提取方法 canny/sobel/hed/yoloe26")
    ap.add_argument("--shards", type=int, default=0, help="<=0 = 全部 shard（默认全量）")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help=">0 = 只提前 N 张（小批量验证）")
    args = ap.parse_args()

    out_dir = DATASETS_DIR / "contour" / f"imagenet_{args.split}_{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_paths, index = build_index(args.split, args.shards)
    if args.limit and args.limit > 0:
        index = index[: args.limit]
    print(f"[extract] {len(index)} images -> {out_dir} method={args.method} size={args.size} workers={args.workers}", flush=True)

    done = 0
    with Pool(processes=args.workers, initializer=_init, initargs=(file_paths, args.method, args.size)) as pool:
        for n in pool.imap_unordered(_one, _tasks(file_paths, index, str(out_dir), args.size), chunksize=16):
            done += n
            if done % 2000 == 0:
                print(f"  [extract] {done}/{len(index)}", flush=True)

    manifest = {
        "source_name": f"imagenet_{args.split}",
        "method": args.method,
        "size": args.size,
        "frame_count": len(index),
        "frames_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[extract] done: {done} new frames, manifest written -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
