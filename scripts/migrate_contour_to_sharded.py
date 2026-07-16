#!/usr/bin/env python3
"""把 contour 目录里**根目录下**的扁平 frame_*.png 迁到分桶子目录。

针对历史扁平布局（单目录百万文件导致 NTFS 退化）→ 转成 <i//5000:04d>/frame_*.png
（每子目录 ≤5000，与 extract_imagenet_contour.py:_frame_path 一致）。

- 只扫根目录（os.scandir，不递归）—— 已在子目录里的不动。
- 同卷 os.rename = 元数据移动（快，不复制）。
- 幂等：再跑无操作（根目录已无 frame_*.png）。
- 去重：目标已存在则删根副本。
- 跑完刷新 manifest.json 的 frame_count = 实际 PNG 数（修脏值）。

用法：
  python scripts/migrate_contour_to_sharded.py --dir datasets/contour/imagenet_train_hed
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

_BUCKET_SIZE = 5000
_FRAME_RE = re.compile(r"^frame_(\d+)\.png$")


def count_pngs(root: Path) -> int:
    n = 0
    for _, _, fs in os.walk(root):
        n += sum(1 for f in fs if f.endswith(".png"))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="contour 目录（如 datasets/contour/imagenet_train_hed）")
    ap.add_argument("--bucket-size", type=int, default=_BUCKET_SIZE)
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")
    bs = args.bucket_size

    moved = 0
    dups = 0
    scanned = 0
    import time
    t0 = time.time()
    # 只扫根目录条目（不递归，避免遍历已分桶的子目录）
    for entry in os.scandir(root):
        scanned += 1
        if not entry.is_file():
            continue
        m = _FRAME_RE.match(entry.name)
        if not m:
            continue  # manifest.json / 其它
        i = int(m.group(1))
        bucket = root / f"{i // bs:04d}"
        bucket.mkdir(parents=True, exist_ok=True)
        dst = bucket / entry.name
        src = Path(entry.path)
        if dst.exists():
            # 目标已存在（重复）→ 删根副本
            try:
                os.remove(src)
                dups += 1
            except OSError:
                pass
            continue
        try:
            os.rename(src, dst)
            moved += 1
        except OSError as e:
            print(f"  rename fail {src} -> {dst}: {e}", flush=True)
        if (moved + dups) % 50000 == 0 and (moved + dups) > 0:
            print(f"  [migrate] moved={moved} dups={dups} scanned={scanned} ({(time.time()-t0)/60:.1f}min)", flush=True)

    # 刷新 manifest frame_count = 实际 PNG 数
    total = count_pngs(root)
    mf = root / "manifest.json"
    if mf.is_file():
        try:
            md = json.loads(mf.read_text(encoding="utf-8"))
            md["frame_count"] = total
            if "frames_dir" not in md:
                md["frames_dir"] = str(root)
            mf.write_text(json.dumps(md, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            print(f"  manifest refresh fail: {e}", flush=True)

    print(f"[migrate] done: moved={moved} dups={dups} total_pngs={total} in {(time.time()-t0)/60:.1f}min; manifest.frame_count={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
