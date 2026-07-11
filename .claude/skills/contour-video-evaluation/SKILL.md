---
name: contour-video-evaluation
description: |
  Contour-Video 轮廓视频压缩评测库(benchmark/video/)使用指南。用于两阶段轮廓视频压缩评测:阶段1提取轮廓帧、阶段2用标准视频 codec 压缩并评测。
  触发场景：(1) 从原始视频提取轮廓视频(阶段1) (2) 跑视频 codec 压缩评测(阶段2) (3) 跑 verify 端到端自检 (4) 查看/调整 results.json 结果格式 (5) 增删 codec 或提取器/调整 CRF
---

# Contour-Video 轮廓视频压缩评测库

本 skill 提供 `benchmark/video/` 库的完整使用与基准测试指南。库位于**仓库根**下的 `benchmark/video/`(infraredComp 项目)。所有路径相对仓库根;先 `cd` 到含 `pyproject.toml` 的仓库根再运行命令,或用 `$(git rev-parse --show-toplevel)` 定位。

## 项目结构

```text
infraredComp/benchmark/video/
├── config.py            # 路径常量(datasets/raw|contour, results/video)
├── ffmpeg_util.py       # ffmpeg/ffprobe 发现 + run_ffmpeg + probe
├── data.py              # ContourArtifact / VideoCompressionResult 数据类
├── extractors/          # 阶段1 可插拔提取器
│   ├── base.py          # ContourExtractor ABC + EXTRACTOR_REGISTRY + register/build_extractor
│   ├── canny.py         # @register("canny")
│   └── sobel.py         # @register("sobel")
├── stage1_extract.py    # resolve_input/demux_to_frames/extract_contour_video/load_contour_frames
├── codecs/              # 阶段2 视频 codec
│   ├── base.py          # VideoCodec ABC + CODEC_REGISTRY + build_codec
│   ├── x264.py x265.py svtav1.py vp9.py   # 各 @register_codec
├── stage2_benchmark.py  # benchmark_codec / run_benchmark / save_results_json
├── metrics.py           # 复用 benchmark.metrics + per_frame_quality/temporal_consistency/fps_from_timed
├── aggregate.py         # 唯一共享聚合器(snake_case) + aggregate_rd_curve + bests
├── visualize.py         # 图表 + CSV/Markdown
├── html_report.py       # report.html(模板 video_report_template.html)
├── artifact_io.py      # load_artifact(从 manifest 重建 ContourArtifact)
├── __main__.py          # CLI
└── verify.py            # 端到端自检(合成小视频)
```

## 安装

```bash
cd <repo-root>                          # 含 pyproject.toml 的仓库根
uv sync                                   # 安装依赖(含 torch/compressai/opencv)
ffmpeg -version                           # 需带 libx264/libx265/libsvtav1/libvpx
```

数据目录约定:`datasets/raw/`(用户原始视频)、`datasets/contour/<source>/`(阶段1 产物)、`results/video/`(阶段2 产物)。`datasets/` 整目录在 `.gitignore` 内(大数据本地存),不入库。

## 0. 数据集准备

`datasets/raw/` 里的视频由下载脚本准备(脚本本身入库,产物不入库):

| 脚本 | 数据集 | 产出 |
|------|--------|------|
| `scripts/download_osu_color_thermal.py` | OTCBVS Dataset 03(OSU Color-Thermal)热红外 | `datasets/raw/osu_color_thermal/seq{1..6}.mp4` + `manifest.json` |
| `scripts/download_dataset.py` | FLIR ADAS(红外,via kagglehub) | `datasets/FLIR_ADAS_1_3/` |

```bash
# OSU Color-Thermal:6 段热红外序列(320×240,h264/yuv420p/25fps)
python3 scripts/download_osu_color_thermal.py            # 幂等;--force 重下;--dry-run 预览
# 跑评测:每段单独喂 stage1
uv run python -m benchmark.video --input datasets/raw/osu_color_thermal/seq1.mp4 \
  --method canny --crfs 18,23,28,33
```

baseline 选型与样本量说明见博客 `infrared-compression-datasets-survey`:OSU Color-Thermal 6 段够做方向性 baseline,正式结论建议扩到 ~12-16 段(补 BU-TIV / FLIR ADAS)。

## 1. 阶段1 提取轮廓视频

```bash
# 从 mp4 提取 canny 轮廓,限 30 帧
uv run python -m benchmark.video --input datasets/raw/xxx.mp4 --method canny --frames 30 --extract-only

# 从帧目录提取 sobel 轮廓
uv run python -m benchmark.video --input datasets/raw/frames_dir --method sobel --extract-only
```

产出:`datasets/contour/<source_name>/frame_%06d.png`(无损灰度)+ `manifest.json`。

| manifest 字段 | 类型 | 说明 |
|------|------|------|
| source_name | str | 原始视频名(stem) |
| method | str | 提取器名(canny/sobel) |
| frame_count | int | 帧数 |
| fps | float | 视频帧率(帧目录默认 25,可用 `--fps` 覆盖) |
| width/height | int | 轮廓帧尺寸 |
| duration_s | float | 时长 |

新增提取器:在 `extractors/` 加一个 `@register("name")` 的 `ContourExtractor` 子类,实现 `extract(frame_gray)->uint8`。

## 2. 阶段2 压缩评测

```bash
# 全流程:提取 + 压缩评测(4 codec × 4 CRF)
uv run python -m benchmark.video --input datasets/raw/xxx.mp4 \
  --method canny --crfs 18,23,28,33 --codecs x264,x265,svtav1,vp9

# 仅阶段2:复用已有轮廓视频(--skip-extract 指向 contour 目录)
uv run python -m benchmark.video --input datasets/contour/demo --skip-extract \
  --crfs 23,28 --codecs x264,vp9
```

支持的 codec:`x264`(h264/libx264)、`x265`(hevc/libx265)、`svtav1`(av1/libsvtav1)、`vp9`(vp9/libvpx-vp9)。
CRF 范围 0-63(数值越大质量越低、码率越低);AV1 较慢,用 `--frames` 限帧。

## 3. 结果格式(results/video/results.json)

```json
{ "generated_at": "2026-07-11T...", "runs": [ VideoCompressionResult, ... ] }
```

| VideoCompressionResult 字段 | 类型 | 说明 |
|------|------|------|
| id | str | `{sequence}|{codec}|crf{crf}` |
| codec / codec_family / crf | str/str/int | codec 名/族/CRF |
| sequence_name / method | str/str | 轮廓视频名/提取器 |
| frame_count / fps / width / height | int/float/int/int | 序列信息 |
| psnr / ssim | float/float | 全序列逐帧均值(dB / [0,1]) |
| per_frame_psnr / per_frame_ssim | list[float] | 逐帧指标 |
| bitrate_kbps / bpp / compression_ratio | float | 码率 / 每像素比特 / 压缩比 |
| compressed_bytes / duration_s | int/float | 码流字节 / 时长 |
| encode_time_ms / decode_time_ms | float | 编解码耗时(ms,wall-clock) |
| enc_fps / dec_fps | float | 编解码帧率 |
| temporal_metric | float | 逐帧 PSNR 标准差(越小越稳) |
| decoded_sample | str | 一帧重建图路径 |

## 4. 端到端自检

```bash
uv run python -m benchmark.video.verify
# 合成 even(64×64)+ odd(65×63)视频 → canny → x264/vp9@crf23 → 校验指标/产物 → ALL PASS
```

## 5. 增删 codec / 提取器

- 新 codec:在 `codecs/` 加 `@register_codec("name")` 的 `VideoCodec` 子类,设 `encoder`(ffmpeg 编码器名)/`family`/`default_preset`/`ext`,按需重写 `encode_args`/`decode_args`。基类已统一 `-pix_fmt yuv420p` + 奇数尺寸 `pad`。
- 新提取器:`extractors/` 加 `@register("name")` 子类,实现 `extract(frame_gray)->uint8`。
- 均通过注册表自动发现,无需改 CLI。

## 关键约定

- **统一 `-pix_fmt yuv420p`**:所有 codec 编码统一像素格式(chroma 一致、PSNR 可比),decode 用 `gray` 出单通道 PNG。绝不让各 codec pix_fmt 不同。
- **奇数尺寸**:`encode_args` 自动加 `pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black`,重建帧裁回原 W×H 再算指标。
- **ground truth 单一读取路径**:`stage1_extract.load_contour_frames(artifact)` 用 `cv2.imread(IMREAD_GRAYSCALE)`。
- **两阶段解耦**:阶段1 产物(无损 PNG)既是阶段2 输入也是质量基准;`--extract-only`/`--skip-extract` 可独立运行各阶段。

## 常用命令

```bash
# 自检
uv run python -m benchmark.video.verify

# 全流程评测
uv run python -m benchmark.video --input datasets/raw/xxx.mp4 --method canny --crfs 18,23,28,33

# 仅阶段1
uv run python -m benchmark.video --input datasets/raw/xxx.mp4 --method sobel --extract-only

# 仅阶段2(复用 contour)
uv run python -m benchmark.video --input datasets/contour/demo --skip-extract --codecs x264,vp9

# 查看结果
cat results/video/results.json | python3 -m json.tool | head -40
ls results/video/charts/
```
