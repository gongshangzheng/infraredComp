#!/usr/bin/env python3
"""把 datasets/ 重排成 datasets/{method}/{dataset}/ 布局 + 旧路径留兼容 junction。

旧布局 → 新布局：
  原始: datasets/{BSDS500,imagenet,SACo-VEval}（junction）、datasets/FLIR_ADAS_1_3（真实）、
        datasets/raw/{xiph_cif,osu_color_thermal} → datasets/original/<name>/
  轮廓视频（2 层级）: datasets/contour/<src>/<method>/ → datasets/<method>/<src>/
  轮廓图像（方法在名里）: datasets/contour/imagenet_<split>_<method>/ → datasets/<method>/imagenet_<split>/
                        datasets/contour/bsds_<split>_gt/        → datasets/gt/bsds_<split>/

每个搬走的目录在旧路径留一个 junction 指新路径（兼容在跑训练 + 渐进迁移代码），最后用
--remove-compat 清。

用法：
  python scripts/reorg_datasets.py --dry-run      # 预览搬迁计划
  python scripts/reorg_datasets.py                # 执行搬迁 + 留兼容 junction
  python scripts/reorg_datasets.py --remove-compat  # 清兼容 junction（代码全迁完后）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from benchmark.video.config import DATASETS_DIR  # noqa: E402

METHODS = {"canny", "sobel", "hed", "pidinet", "yoloe26", "gt"}
RAW_TOP = {"BSDS500", "imagenet", "SACo-VEval", "FLIR_ADAS_1_3"}  # 顶层原始（junction/真实）


def _is_junction(p: Path) -> bool:
    """Windows junction / symlink（reparse point）。"""
    return p.is_symlink() or (os.name == "nt" and p.exists() and bool(os.stat(p).st_file_attributes & 0x400))


def _make_junction(link: Path, target: Path) -> None:
    """Windows junction link → target（无需管理员）。非 Windows 退回 symlink。"""
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())],
                       check=True, capture_output=True)
    else:
        os.symlink(str(target.resolve()), str(link))


def _remove_link(p: Path) -> None:
    if _is_junction(p):
        if os.name == "nt":
            os.rmdir(p)  # junction 用 rmdir 删链接不删 target
        else:
            os.unlink(p)
    elif p.is_dir():
        import shutil
        shutil.rmtree(p)


def _move(src: Path, dst: Path, dry: bool, compat: bool) -> str:
    """把 src 搬到 dst；dst 存在则 skip；可选在 src 留兼容 junction → dst。"""
    if not src.exists() and not _is_junction(src):
        return f"SKIP (missing): {src}"
    if dst.exists():
        return f"SKIP (exists): {dst}"
    if dry:
        return f"DRY  move: {src} -> {dst}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_junction(src):
        target = Path(os.readlink(src)) if src.is_symlink() else Path(_junction_target(src))
        _make_junction(dst, target)  # 新路径重建 junction → 原 target
        _remove_link(src)
    else:
        os.rename(src, dst)
    if compat and not src.exists():
        _make_junction(src, dst)  # 旧路径留兼容 junction → 新
    return f"move: {src} -> {dst}" + (" (+compat)" if compat else "")


def _junction_target(p: Path) -> str:
    """Windows junction 的真实 target（realpath）。"""
    return str(Path(p).resolve(strict=False))


def _parse_flat_contour(name: str):
    """imagenet_<split>_<method> / bsds_<split>_gt → (method, dataset) 或 None。"""
    toks = name.split("_")
    if len(toks) >= 3 and toks[-1] in METHODS:
        return toks[-1], "_".join(toks[:-1])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remove-compat", dest="remove_compat", action="store_true",
                    help="清旧路径的兼容 junction（代码全迁完后）")
    args = ap.parse_args()

    if not DATASETS_DIR.is_dir():
        print(f"[reorg] datasets 目录不存在: {DATASETS_DIR}")
        return 1
    print(f"[reorg] DATASETS_DIR={DATASETS_DIR}  dry={args.dry_run}  remove_compat={args.remove_compat}\n")

    if args.remove_compat:
        # 清旧路径兼容 junction：contour/* 与顶层 raw junction 与 raw/
        n = 0
        for old in [DATASETS_DIR / "contour", DATASETS_DIR / "raw", *(
            DATASETS_DIR / r for r in RAW_TOP)]:
            if old.exists() and _is_junction(old):
                print(f"[reorg] rm compat junction: {old}")
                if not args.dry_run:
                    _remove_link(old)
                n += 1
            elif old.is_dir():
                # contour/ raw/ 下可能还有兼容 junction 子项
                for sub in old.iterdir():
                    if _is_junction(sub):
                        print(f"[reorg] rm compat junction: {sub}")
                        if not args.dry_run:
                            _remove_link(sub)
                        n += 1
        print(f"\n[reorg] removed {n} compat junctions")
        return 0

    plan = []

    # 1. 原始：顶层 junction/真实 + raw/* → original/<name>/
    for name in RAW_TOP:
        plan.append(_move(DATASETS_DIR / name, DATASETS_DIR / "original" / name, args.dry_run, compat=True))
    raw_dir = DATASETS_DIR / "raw"
    if raw_dir.is_dir():
        for sub in sorted(raw_dir.iterdir()):
            if sub.name in (".gitkeep",):
                continue
            plan.append(_move(sub, DATASETS_DIR / "original" / sub.name, args.dry_run, compat=True))

    # 2. 轮廓：contour/ 下逐项
    contour_dir = DATASETS_DIR / "contour"
    if contour_dir.is_dir():
        for child in sorted(contour_dir.iterdir()):
            if child.name in (".gitkeep",):
                continue
            # 2a. 视频源（有 method 子目录）→ 每个 method 子目录搬到 <method>/<src>/
            method_subdirs = [s for s in child.iterdir() if s.is_dir() and s.name in METHODS] if child.is_dir() else []
            if method_subdirs:
                for ms in method_subdirs:
                    plan.append(_move(ms, DATASETS_DIR / ms.name / child.name, args.dry_run, compat=True))
            else:
                # 2b. 图像 flat（方法在名里）→ <method>/<dataset>/
                parsed = _parse_flat_contour(child.name)
                if parsed:
                    method, dataset = parsed
                    plan.append(_move(child, DATASETS_DIR / method / dataset, args.dry_run, compat=True))
                else:
                    plan.append(f"SKIP (unknown contour): {child}")

    for line in plan:
        print("  " + line)
    print(f"\n[reorg] {'DRY-RUN' if args.dry_run else 'done'}: {len(plan)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
