#!/usr/bin/env python3
"""一次性预提取 imagenet 边缘到 PNG（落地），供训练快速读取（GPU 喂得满）。

遍历指定 split 的全部 parquet 行（默认全量），用给定方法（hed/canny/sobel）
提取边缘，resize 到 `--size` 存为 frame_XXXXXXX.png + manifest.json。
多进程分片加速；skip-if-exists 可断点续跑。

关键：以 **row group 为工作单元**（每个 ~101MB 组只读一次，处理组内全部 ~871 行），
避免逐图随机读组导致的 thrash（否则会慢 ~50×）。产物命名按全局行号，续跑稳定。

产物目录：datasets/contour/imagenet_<split>_<method>/frame_XXXXXXX.png
训练侧 resolve_training_dataset 检测到该目录有 manifest 即改用 ContourPNGDataset。

用法：
  python scripts/extract_imagenet_contour.py --split train --method hed --workers 16
  python scripts/extract_imagenet_contour.py --split train --method hed --limit 4096   # 小批量验证
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
from benchmark.video.config import raw_dir, contour_dir  # noqa: E402

DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(REPO / "datasets")))
_SPLIT_PREFIX = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def _frame_path(out_dir: str, i: int) -> Path:
    """分桶存盘：<out_dir>/<i//5000:04d>/frame_<i:07d>.png。
    每子目录 ≤5000 文件，避免单目录百万文件导致 NTFS 写入/索引退化。
    ContourPNGDataset 用 rglob 递归读，兼容分桶与扁平两种布局。"""
    bucket = Path(out_dir) / f"{i // 5000:04d}"
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket / f"frame_{i:07d}.png"


def build_groups(split: str, shards: int) -> tuple[list[str], list[tuple[int, int, int, int]], int]:
    """返回 (file_paths, groups, total)。
    groups = [(start_idx, file_idx, row_group, num_rows), ...]，按 parquet 顺序连续编号。
    """
    import pyarrow.parquet as pq
    data_dir = raw_dir("imagenet") / "data"
    prefix = _SPLIT_PREFIX.get(split, split)
    files = sorted(data_dir.glob(f"{prefix}-*.parquet"))
    if not files:
        raise RuntimeError(f"无 imagenet parquet: {data_dir}/{prefix}-*.parquet")
    if shards and shards > 0:
        files = files[: max(1, int(shards))]
    file_paths = [str(f) for f in files]
    groups: list[tuple[int, int, int, int]] = []
    idx = 0
    for fi, fp in enumerate(file_paths):
        md = pq.ParquetFile(fp).metadata
        for rg in range(md.num_row_groups):
            n = md.row_group(rg).num_rows
            groups.append((idx, fi, rg, n))
            idx += n
    return file_paths, groups, idx


# ---- worker 进程状态（spawn 下每进程独立建 extractor / ParquetFile）----------- #
_W: dict = {}


def _init(file_paths: list[str], method: str, size: int, extract_size: int) -> None:
    import cv2  # noqa: F401
    import pyarrow.parquet as pq
    from benchmark.video.extractors import build_extractor
    # cv2.dnn/OpenCV 默认占满全部核，多 worker 下会严重超订（8w×24核=192 路）。
    # 每个 worker 锁单线程，靠 worker 数做并行。
    cv2.setNumThreads(1)
    _W["pq"] = [pq.ParquetFile(fp) for fp in file_paths]
    _W["ex"] = build_extractor(method)
    _W["size"] = size
    _W["extract_size"] = extract_size


def _one_group(task) -> int:
    """读一个 row group 的 image 列一次，处理组内全部行。返回本组新增帧数。"""
    start_idx, fi, rg, nrows, out_dir, size = task
    ex = _W["ex"]
    extract_size = _W["extract_size"]
    col = _W["pq"][fi].read_row_group(rg, columns=["image"]).column("image")  # 整组只读一次
    new = 0
    for r in range(nrows):
        i = start_idx + r
        out = _frame_path(out_dir, i)
        if out.exists():
            continue  # skip-if-exists（续跑）
        cell = col[r].as_py()
        b = cell["bytes"] if isinstance(cell, dict) else cell
        img = Image.open(io.BytesIO(b)).convert("RGB")
        # 先把输入缩到 max 边 = extract_size（bounds hed 成本；最终输出是 size=128，
        # 原生 4K 图提取边缘再缩 128 没意义且慢 ~10×）
        if extract_size and max(img.size) > extract_size:
            ratio = extract_size / max(img.size)
            img = img.resize((max(1, int(img.size[0] * ratio)), max(1, int(img.size[1] * ratio))), Image.BILINEAR)
        # color BGR (extractors expect cv2 BGR; hed/pidinet/yoloe26 trained on color)
        arr = np.ascontiguousarray(np.array(img, dtype=np.uint8)[..., ::-1])
        edges = ex.extract(arr)
        if edges.dtype != np.uint8:
            edges = edges.astype(np.uint8)
        eimg = Image.fromarray(edges)
        if eimg.size != (size, size):  # PIL size=(W,H)
            eimg = eimg.resize((size, size), Image.BILINEAR)
        eimg.save(out)
        new += 1
    return new


def _gen_tasks(groups, out_dir, size, limit):
    acc = 0
    for (start_idx, fi, rg, n) in groups:
        if limit and acc >= limit:
            break
        # 若有限量，截断本组行数
        nrows = n if (not limit or acc + n <= limit) else max(0, limit - acc)
        if nrows > 0:
            yield (start_idx, fi, rg, nrows, out_dir, size)
        acc += n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--method", default="hed", help="轮廓提取方法 canny/sobel/hed/yoloe26")
    ap.add_argument("--shards", type=int, default=0, help="<=0 = 全部 shard（默认全量）")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--extract-size", dest="extract_size", type=int, default=256,
                    help="hed 在该 max 边长上跑（缩输入省算力；最终存 size）。默认 256")
    ap.add_argument("--limit", type=int, default=0, help=">0 = 只提前 N 张（小批量验证）")
    args = ap.parse_args()

    out_dir = contour_dir(args.method, f"imagenet_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    file_paths, groups, total = build_groups(args.split, args.shards)
    target = total if not (args.limit and args.limit > 0) else min(args.limit, total)
    print(f"[extract] target={target} rows (full={total}) method={args.method} size={args.size} workers={args.workers}", flush=True)

    done = 0
    tasks = _gen_tasks(groups, str(out_dir), args.size, args.limit)
    with Pool(processes=args.workers, initializer=_init,
              initargs=(file_paths, args.method, args.size, args.extract_size)) as pool:
        for new in pool.imap_unordered(_one_group, tasks, chunksize=1):
            done += new
            if done and done % 5000 == 0:
                print(f"  [extract] {done}/{target} ({100*done/max(1,target):.1f}%)", flush=True)

    manifest = {
        "source_name": f"imagenet_{args.split}",
        "method": args.method,
        "size": args.size,
        "frame_count": target,
        "frames_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[extract] done: {done} new frames (target {target}), manifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
