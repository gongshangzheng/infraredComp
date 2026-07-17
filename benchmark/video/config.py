"""Path constants for the contour-video benchmark.

Mirrors ProjFlow's single-source-of-truth config pattern: every path lives here,
no other module hardcodes filesystem locations.
"""

import os
from pathlib import Path

# /Users/zhengxinyu/infraredComp/benchmark/video/../../  -> repo root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Datasets 树位置可经环境变量配置:大数据集可移出仓库(换盘/省备份),默认 <repo>/datasets。
# 布局：datasets/raw/<dataset>/（原始数据;大数据集为 D:/data 软链接）+ datasets/<method>/<dataset>/（轮廓产物:canny/sobel/hed/pidinet/yoloe26/gt）。
# raw_dir()/contour_dir() 是单一来源,勿在业务代码硬编码 datasets/... 字面。
DATASETS_DIR = Path(os.getenv("INFRACOMP_DATASETS_DIR", str(BASE_DIR / "datasets")))
RAW_DIR = DATASETS_DIR / "raw"            # canonical raw root(原始数据集)
CONTOUR_DIR = DATASETS_DIR / "contour"    # [deprecated 兼容] 旧 contour/,阶段4 删;新代码用 contour_dir()


def raw_dir(dataset: str) -> Path:
    """原始数据集目录:datasets/raw/<dataset>/(如 raw/xiph_cif、raw/BSDS500;大数据集为 D:/data 软链接)。"""
    return DATASETS_DIR / "raw" / dataset


def contour_dir(method: str, dataset: str) -> Path:
    """轮廓产物目录：datasets/<method>/<dataset>/（如 canny/akiyo_cif、gt/bsds_train、hed/imagenet_train）。"""
    return DATASETS_DIR / method / dataset

RESULTS_DIR = BASE_DIR / "results" / "video"
BITSTREAMS_DIR = RESULTS_DIR / "bitstreams"
RECON_DIR = RESULTS_DIR / "recon"
CHARTS_DIR = RESULTS_DIR / "charts"
RESULTS_JSON = RESULTS_DIR / "results.json"

# Pixel format contract: every codec encodes with yuv420p (portable, consistent
# chroma subsampling so PSNR is comparable across codecs), and decodes back to
# single-channel gray PNG. Never let per-codec pix_fmt differ.
ENCODE_PIX_FMT = "yuv420p"
DECODE_PIX_FMT = "gray"


def ensure_dirs() -> None:
    """Create the output directories (idempotent)."""
    for d in (RAW_DIR, CONTOUR_DIR, RESULTS_DIR, BITSTREAMS_DIR, RECON_DIR, CHARTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
