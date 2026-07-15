#!/usr/bin/env python3
"""把 BSDS500 的 GT（.mat 边缘标注）转成 PNG（软边缘目标）。

BSDS500 layout（datasets/BSDS500，junction → D:/data/BSDS500）：
  images/{train,val,test}/*.jpg + groundTruth/{train,val,test}/*.mat
.mat 里 `groundTruth` 是 (1,N) object 数组（N=5-6 标注者），每项 struct 字段
('Segmentation','Boundaries')；Boundaries 是 uint8 HxW {0,1} 的二值边缘图。

转换：平均所有标注者的 Boundaries → soft ∈[0,1] → ×255 uint8 L PNG，存到
datasets/contour/bsds_<split>_gt/frame_<i:07d>.png + manifest.json。
flat 布局（split 仅 ~200 张，不分桶）；frame_ 前缀让 ContourPNGDataset rglob 能读。

用法：
  python scripts/convert_bsds_gt.py                  # train/val/test 全转
  python scripts/convert_bsds_gt.py --splits train   # 只转 train
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(REPO / "datasets")))


def _mat_to_soft_edge(mat_path: Path) -> np.ndarray:
    """load .mat → 平均所有标注者的 Boundaries → uint8 HxW 软边缘。"""
    import scipy.io
    m = scipy.io.loadmat(str(mat_path))
    gt = m["groundTruth"]                      # (1, N) object array
    n = gt.shape[1]
    acc = None
    for k in range(n):
        b = gt[0, k]["Boundaries"][0, 0]        # uint8 HxW {0,1}
        b = b.astype(np.float32)
        acc = b.copy() if acc is None else acc + b
    soft = acc / n                             # [0,1]
    return (np.clip(soft, 0.0, 1.0) * 255.0).astype(np.uint8)


def convert_split(src: Path, out_root: Path, split: str, save_size: int) -> int:
    gt_dir = src / "groundTruth" / split
    mats = sorted(gt_dir.glob("*.mat"))
    if not mats:
        print(f"[bsds-gt] 无 .mat：{gt_dir}", flush=True)
        return 0
    out_dir = out_root / f"bsds_{split}_gt"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, mp in enumerate(mats):
        out = out_dir / f"frame_{i:07d}.png"
        if out.exists():
            n += 1
            continue  # skip-if-exists
        edge = _mat_to_soft_edge(mp)           # HxW uint8
        img = Image.fromarray(edge, mode="L")
        if save_size and save_size > 0 and img.size != (save_size, save_size):
            img = img.resize((save_size, save_size), Image.BILINEAR)
        img.save(out)
        n += 1
    manifest = {
        "source_name": f"bsds_{split}",
        "method": "gt",
        "size": save_size,
        "frame_count": n,
        "frames_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bsds-gt] {split}: {n} frames -> {out_dir}", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DATASETS_DIR / "BSDS500"))
    ap.add_argument("--out-root", default=str(DATASETS_DIR / "contour"))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--save-size", type=int, default=0, help="0=原生分辨率；>0 则 resize 到该尺寸存盘")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"BSDS500 源目录不存在：{src}（先建 datasets/BSDS500 junction）")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in args.splits.split(","):
        split = split.strip()
        if split:
            total += convert_split(src, out_root, split, args.save_size)
    print(f"[bsds-gt] done: {total} frames total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
