import os

# 仓库根目录(server/ 的上一级)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 各模块路径
MANAGEMENT_DIR = os.path.join(BASE_DIR, "management")
PAPERS_DIR = os.path.join(BASE_DIR, "papers")

# 轮廓视频评测模块路径
# Datasets 树位置可经环境变量配置:大数据集可移出仓库,默认 <repo>/datasets。
# CONTOUR_DIR 由 DATASETS_DIR 派生,自动跟随;/runs 端点据此列出 contour 目录。
DATASETS_DIR = os.environ.get("INFRACOMP_DATASETS_DIR", os.path.join(BASE_DIR, "datasets"))
CONTOUR_DIR = os.path.join(DATASETS_DIR, "contour")  # 阶段1 产出的无损轮廓帧
RESULTS_VIDEO_DIR = os.path.join(BASE_DIR, "results", "video")
RESULTS_VIDEO_JSON = os.path.join(RESULTS_VIDEO_DIR, "results.json")
# 评测输出根目录（压缩码流 bitstreams/ + 重建视频 recon/），供 /api/evaluation/outputs 按需服务
OUTPUTS_DIR = RESULTS_VIDEO_DIR

# 训练模块路径（镜像 results/video/）
TRAINING_DIR = os.path.join(BASE_DIR, "results", "training")
TRAINING_METRICS_JSON = os.path.join(TRAINING_DIR, "metrics.json")
CHECKPOINTS_DIR = os.path.join(TRAINING_DIR, "checkpoints")
TRAINING_LOGS_DIR = os.path.join(TRAINING_DIR, "logs")
TRAINING_OUTPUTS_DIR = TRAINING_DIR

# 论文数据库路径(本地 SQLite)
PAPERS_DB = os.path.join(BASE_DIR, "data", "papers.db")

# 论文缩略图缓存(arXiv PDF 首页 → WebP)
PAPERS_CACHE_DIR = os.path.join(PAPERS_DIR, "cache")
THUMBNAILS_DIR = os.path.join(PAPERS_CACHE_DIR, "thumbnails")

# CORS 配置(infraredComp Vite dev 3001;保留 3000/5173 以备调整)
CORS_ORIGINS = [
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
