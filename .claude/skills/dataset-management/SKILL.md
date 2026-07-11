---
name: dataset-management
description: Manage datasets in infraredComp — configure dataset location via INFRACOMP_DATASETS_DIR, download the FLIR ADAS dataset from Kaggle, and add raw/contour video inputs for the contour-video benchmark. Use when obtaining or configuring datasets, or before running the benchmark.
---

# datasets 管理

说明 infraredComp 中如何获取、配置与接入数据集。完整索引见 `datasets/README.md`(tracked)。

## 数据集位置

根目录可经 `INFRACOMP_DATASETS_DIR` 重定位(默认 `<repo>/datasets`):

```bash
export INFRACOMP_DATASETS_DIR=/data/infrared
```

被 `benchmark/video/config.py`、`server/config.py`、`scripts/download_dataset.py`、`benchmark/runner.py`、`benchmark/demo.py` 读取。路径常量遵循**单一来源**约定,勿在业务代码硬编码。

## 两个数据域(勿混为一谈)

1. **FLIR ADAS 1.3**(红外图像,legacy 图像 benchmark):从 Kaggle 下载,`thermal_16_bit` TIFF 作为压缩评测输入。许可证见下载后的 PDF,仅用于研究。
2. **raw / contour 视频**(video benchmark):raw = 用户自备视频/帧目录(运行时 `--input` 指定,位置不固定);contour = 阶段1 提取的无损 PNG + `manifest.json`,存 `${INFRACOMP_DATASETS_DIR}/contour/<source>/`。

## 下载 FLIR

需 Kaggle 凭证:`KAGGLE_USERNAME` + `KAGGLE_KEY` 环境变量,或 `~/.kaggle/kaggle.json`(https://www.kaggle.com/settings/account → Create New Token)。

```bash
uv run python scripts/download_dataset.py                 # 最新版
uv run python scripts/download_dataset.py --version 3     # 锁定版本(可复现)
uv run python scripts/download_dataset.py --force         # 覆盖已有目录
```

完成后写 `datasets/manifest.json`(含 version / size_bytes / downloaded_at)。

## 接入新数据集跑 benchmark

`--input` 接受视频文件或帧目录(`benchmark/video/stage1_extract.py::resolve_input`):

```bash
uv run python -m benchmark.video --input /path/to/video.mp4 --method canny
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/FLIR_ADAS_1_3/video/thermal_8_bit --method canny --extract-only
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/contour/demo --skip-extract   # 仅阶段2
```

## git 策略

`datasets/` 下按媒体扩展名忽略(`*.mp4 *.png *.jpg ...`),不依赖固定目录名。`datasets/README.md`、`datasets/manifest.json`、`datasets/contour/*/manifest.json` 被追踪作为索引;大文件不进 git。

## 约定速查

- 路径常量:`benchmark/video/config.py` 与 `server/config.py` 是单一来源。
- 后端只读:`/api/benchmark/runs` 列 `contour/` 目录 + manifest,不读 raw 输入。
- ffmpeg 统一 `-pix_fmt yuv420p` 编码、`gray` 解码(详见 AGENTS.md)。
