#!/usr/bin/env python3
"""GPU 批量预提取 imagenet hed 边缘到 PNG（落地）。

单进程 + GPU 批量推理（cv2.dnn 无 CUDA，CPU 要 ~20h；5090 批量 ~30min）。
以 row group 为读单位（每 ~101MB 组读一次），组内解码成子批喂 GPU，输出 resize 到
save_size 存 PNG。skip-if-exists 可续跑。输出与 cv2 版 hed 对齐（同权重，~0.91 corr）。

用法：
  python scripts/extract_imagenet_hed_gpu.py --split train --batch 64 --extract-size 256
  python scripts/extract_imagenet_hed_gpu.py --split train --limit 4096   # 小批量验证
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from scripts.extract_imagenet_contour import build_groups  # noqa: E402
from scripts.hed_gpu import HEDTorch, load_hed_from_caffe, hed_edges_batch  # noqa: E402

DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(REPO / "datasets")))
_SPLIT = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}


def _frame_path(out_dir: Path, i: int) -> Path:
    """分桶存盘：<out_dir>/<i//5000:04d>/frame_<i:07d>.png（每子目录 ≤5000，避免 NTFS 单目录退化）。"""
    bucket = out_dir / f"{i // 5000:04d}"
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket / f"frame_{i:07d}.png"


def _frame_exists(out_dir: Path, i: int) -> bool:
    """已存在则跳过：查分桶路径 **或** 旧扁平路径（兼容历史扁平布局/未迁移完）。不 mkdir。"""
    bucket = out_dir / f"{i // 5000:04d}"
    if (bucket / f"frame_{i:07d}.png").exists():
        return True
    return (out_dir / f"frame_{i:07d}.png").exists()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--extract-size", dest="extract_size", type=int, default=256, help="hed 推理分辨率")
    ap.add_argument("--save-size", dest="save_size", type=int, default=128, help="存盘 PNG 分辨率")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = DATASETS_DIR / "contour" / f"imagenet_{args.split}_hed"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    file_paths, groups, total = build_groups(args.split, args.shards)
    target = total if not (args.limit and args.limit > 0) else min(args.limit, total)
    print(f"[hed-gpu] target={target} (full={total}) extract_size={args.extract_size} "
          f"save_size={args.save_size} batch={args.batch} dev={dev}", flush=True)

    model = HEDTorch().to(dev)
    load_hed_from_caffe(model)
    model.eval()

    import pyarrow.parquet as pq
    pfs = [pq.ParquetFile(fp) for fp in file_paths]

    def flush(imgs, idxs):
        if not imgs:
            return 0
        edges = hed_edges_batch(model, imgs, size=args.extract_size, device=dev)  # (N,es,es) uint8
        n = 0
        for k, i in enumerate(idxs):
            e = edges[k]
            if e.shape[0] != args.save_size:
                e = np.array(Image.fromarray(e).resize((args.save_size, args.save_size), Image.BILINEAR))
            Image.fromarray(e).save(_frame_path(out_dir, i))
            n += 1
        return n

    done = 0
    t0 = time.time()
    acc_rows = 0
    for (start_idx, fi, rg, nrows) in groups:
        if args.limit and acc_rows >= args.limit:
            break
        col = pfs[fi].read_row_group(rg, columns=["image"]).column("image")  # 整组读一次
        buf_imgs, buf_idx = [], []
        for r in range(nrows):
            if args.limit and acc_rows + r >= args.limit:
                break
            i = start_idx + r
            if _frame_exists(out_dir, i):
                continue
            cell = col[r].as_py()
            b = cell["bytes"] if isinstance(cell, dict) else cell
            buf_imgs.append(np.array(Image.open(io.BytesIO(b)).convert("L")))
            buf_idx.append(i)
            if len(buf_imgs) >= args.batch:
                done += flush(buf_imgs, buf_idx)
                buf_imgs, buf_idx = [], []
                if done and done % 5000 == 0:
                    el = time.time() - t0
                    print(f"  [hed-gpu] {done}/{target} ({100*done/max(1,target):.1f}%) "
                          f"{done/el:.0f} f/s eta {((target-done)/max(1,done/el))/60:.0f}min", flush=True)
        done += flush(buf_imgs, buf_idx)
        acc_rows += nrows

    manifest = {"source_name": f"imagenet_{args.split}", "method": "hed", "size": args.save_size,
                "frame_count": target, "frames_dir": str(out_dir)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[hed-gpu] done: {done} new frames in {(time.time()-t0)/60:.1f}min, manifest -> {out_dir/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
