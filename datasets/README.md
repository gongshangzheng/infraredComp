# datasets/ — 数据集索引

本文件是 **tracked** 的数据集索引。`datasets/` 下的大体积媒体文件(视频/图像帧)**不进 git**(见 `.gitignore`),仅本索引与各 contour 产物的 `manifest.json` 被追踪。新人 clone 后据此获取数据。

## 数据集位置配置

数据集树的根目录可经环境变量重定位(默认 `<repo>/datasets`):

```bash
export INFRACOMP_DATASETS_DIR=/data/infrared   # 把大数据集放到仓库外(换盘/省备份)
```

`benchmark/video/config.py`、`server/config.py`、`scripts/download_dataset.py`、`benchmark/runner.py`、`benchmark/demo.py` 均读取此变量。路径常量遵循**单一来源**约定,业务代码不硬编码路径。

## 两个数据域(请勿混为一谈)

### 1. FLIR ADAS 1.3(红外图像集,legacy 图像 benchmark 用)

| 项 | 值 |
|---|---|
| 来源 | Kaggle [`deepnewbie/flir-thermal-images-dataset`](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset) |
| 体积 | ~15GB |
| 许可证 | FLIR ADAS Dataset V1.3 License(见下载后的 `ADAS User License Agreement (26.Jul.2018) (Final).pdf`);**仅用于研究评测** |
| 获取 | `uv run python scripts/download_dataset.py`(需 Kaggle 凭证,见脚本头部) |
| 用途 | `benchmark/runner.py` / `benchmark/demo.py` 读取 `thermal_16_bit` 下的 16-bit TIFF 作为压缩评测输入 |
| 路径 | `${INFRACOMP_DATASETS_DIR}/FLIR_ADAS_1_3/{train,val,video}/thermal_16_bit` |

### 2. raw / contour 视频(video benchmark 用)

| 项 | 值 |
|---|---|
| raw | 用户自备的输入——视频文件(`.mp4/.avi/...`)或帧目录(`.png/.jpg/...`)。位置不固定,运行时由 `--input` 指定 |
| contour | 阶段1 提取产物:无损灰度 PNG 帧序列 + `manifest.json`,存 `${INFRACOMP_DATASETS_DIR}/contour/<source>/`。可由 raw 重算,故帧 PNG 不进 git,仅 `manifest.json` 追踪 |
| 获取/产生 | `uv run python -m benchmark.video --input <raw> --method canny [--extract-only]` |
| 用途 | 阶段2 压缩评测的输入与质量基准 |

## 接入新数据集

```bash
# 视频
uv run python -m benchmark.video --input /path/to/video.mp4 --method canny

# 帧目录(如 FLIR 的 thermal_8_bit)
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/FLIR_ADAS_1_3/video/thermal_8_bit --method canny --extract-only

# 仅阶段2(复用已有 contour 产物)
uv run python -m benchmark.video --input ${INFRACOMP_DATASETS_DIR}/contour/demo --skip-extract
```

`--input` 接受视频文件或帧目录(`benchmark/video/stage1_extract.py::resolve_input`)。

## git 策略

`datasets/` 下按**媒体扩展名**忽略(`*.mp4 *.png *.jpg *.tif ...`),不依赖固定目录名,以适配可配置的输入位置。`README.md`、`manifest.json`、`contour/*/manifest.json` 等小文件被追踪。
