"""Path constants for the contour-video benchmark.

Mirrors ProjFlow's single-source-of-truth config pattern: every path lives here,
no other module hardcodes filesystem locations.
"""

import os
from pathlib import Path

# /Users/zhengxinyu/infraredComp/benchmark/video/../../  -> repo root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Datasets 树位置可经环境变量配置:大数据集可移出仓库(换盘/省备份),默认 <repo>/datasets。
# RAW_DIR / CONTOUR_DIR 均由 DATASETS_DIR 派生,自动跟随。
DATASETS_DIR = Path(os.getenv("INFRACOMP_DATASETS_DIR", str(BASE_DIR / "datasets")))
RAW_DIR = DATASETS_DIR / "raw"            # user-provided raw videos (mp4/avi)
CONTOUR_DIR = DATASETS_DIR / "contour"    # stage-1 output: lossless PNG + manifest

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
