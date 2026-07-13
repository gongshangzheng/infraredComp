---
name: contour-video-evaluation
description: |
  Contour-Video 轮廓视频压缩评测库(benchmark/video/)使用指南,含两模式评测(speed run 视频网格 / formal test 平均指标)。用于两阶段轮廓视频压缩评测:阶段1提取轮廓帧、阶段2用标准视频 codec 压缩并评测。
  触发场景:(1) 从原始视频提取轮廓视频(阶段1) (2) 跑视频 codec 压缩评测(阶段2,支持 --sequences 子集) (3) speed run(少量视频,视频网格看主观) (4) formal test(全量,per-(codec,crf) 平均指标) (5) 跑 verify 端到端自检 (6) 查看/调整 results 结果格式 / aggregate 端点
---

# Contour-Video 轮廓视频压缩评测库

本 skill 提供 `benchmark/video/` 库 + 评测前端两模式的完整使用指南。库位于仓库根 `benchmark/video/`。所有路径相对仓库根;先 `cd` 到含 `pyproject.toml` 的仓库根再运行命令。

## 项目结构

```text
infraredComp/benchmark/video/
├── config.py            # 路径常量(datasets/raw|contour, results/video)
├── ffmpeg_util.py       # ffmpeg/ffprobe 发现(INFRACOMP_FFMPEG_BIN → PATH → static_ffmpeg fallback)
├── data.py              # ContourArtifact / VideoCompressionResult(含 dataset 字段)
├── extractors/          # 阶段1 可插拔提取器(canny/sobel,@register)
├── stage1_extract.py    # resolve_input/expand_inputs(支持 .y4m + 目录 glob)/extract_contour_video
├── codecs/              # 阶段2 视频 codec(x264/x265/svtav1/vp9,@register)
├── stage2_benchmark.py  # benchmark_codec/run_benchmark/save_results_json(支持 save= + path= + metadata=)
├── metrics.py           # PSNR/SSIM/时序一致性
├── aggregate.py         # aggregate_by_codec + aggregate_by_codec_crf(per-(codec,crf) 平均)+ aggregate_rd_curve + bests
├── repro.py             # build_metadata(git_sha/codecs/crfs/dataset envelope)
├── visualize.py / html_report.py  # 图表 + report.html
├── artifact_io.py      # load_artifact(从 manifest 重建 ContourArtifact)
├── __main__.py          # CLI(--input 可重复 + 目录 glob,多序列累积)
└── verify.py            # 端到端自检(合成小视频)
```

## 安装

```bash
cd <repo-root>
uv sync                                   # 安装依赖(含 torch/compressai/opencv)
# ffmpeg:无需系统安装,`uv add static-ffmpeg` 内置 ffmpeg+ffprobe(static_ffmpeg fallback);
#         或设 INFRACOMP_FFMPEG_BIN 指向 ffmpeg.exe。static-ffmpeg win32 不含 libsvtav1。
```

数据目录:`datasets/raw/`(原始视频)、`datasets/contour/<source>/<method>/`(阶段1 产物,按方法分目录)、`results/video/`(阶段2 产物,每数据集独立 .json)。`datasets/` 大数据 + `results/video/` 运行产物均不入 git(见 `.gitignore`)。

## 0. 数据集准备

| 脚本 | 数据集 | 产出 |
|------|--------|------|
| `scripts/download_xiph_natural.py` | **Xiph derf CIF**(自然视频,6 段 352×288 y4m,可达) | `datasets/raw/xiph_cif/<name>_cif.y4m` + `manifest.json` |
| `scripts/download_osu_color_thermal.py` | OTCBVS Dataset 03(OSU Color-Thermal)热红外(**vcipl-okstate.org 对部分网络 403**) | `datasets/raw/osu_color_thermal/seq{1..6}.mp4` + `manifest.json` |
| `scripts/download_dataset.py` | FLIR ADAS(红外,via kagglehub) | `datasets/FLIR_ADAS_1_3/` |

```bash
# Xiph CIF(推荐,自然视频,公开可达):
uv run python scripts/download_xiph_natural.py            # 6 段 CIF y4m(akiyo/bus/city/flower/foreman/mobile);--force 重下;--dry-run 预览
```

## 1. 阶段1 提取轮廓视频

```bash
# 从 mp4/y4m 提取 canny 轮廓,限 30 帧
uv run python -m benchmark.video --input datasets/raw/xiph_cif/akiyo_cif.y4m --method canny --frames 30 --extract-only

# 整个 xiph_cif 目录(自动 glob *.y4m,多序列累积)
uv run python -m benchmark.video --input datasets/raw/xiph_cif --method canny --extract-only
```

产出:`datasets/contour/<source>/<method>/frame_%06d.png` + `manifest.json`。新增提取器:`extractors/` 加 `@register("name")` 的 `ContourExtractor` 子类,实现 `extract(frame_gray)->uint8`。

## 2. 阶段2 压缩评测

```bash
# 全流程:提取 + 压缩评测(--input 可重复/目录 glob,多序列累积到一个 results.json)
uv run python -m benchmark.video --input datasets/raw/xiph_cif \
  --method canny --crfs 18,23,28,33 --codecs x264,x265,svtav1,vp9

# 仅阶段2(复用 contour,按方法分子目录)
uv run python -m benchmark.video --input datasets/contour/akiyo_cif/canny --skip-extract \
  --crfs 23,28 --codecs x264,vp9
```

codec:`x264`(h264)、`x265`(hevc)、`svtav1`(av1,static-ffmpeg 缺则自动跳过)、`vp9`。CRF 0-63(越大质量越低/码率越低);AV1 慢用 `--frames` 限帧。

## 3. 评测两模式(speed run / formal test)

**评测逻辑统一**(一套 stage1+stage2),差异只在数据集子集 + 展示页。两模式都调同一 baseline 脚本(`run_natural_baseline.py` / `run_osu_baseline.py`),通过 CLI 参数控制:

- **speed run**:少量视频,视频网格看主观。传 `--sequences <stem1,stem2>`(seq 子集)+ 少量 codec/crf。跑完跳 `/evaluation/speed`(视频按 codec 分排,每格 `<video preload=none>` 默认黑屏点击加载,filter 缩范围)。
- **formal test**:全量,平均指标。不传 `--sequences`(全量 seq)+ 全 codec/crf。跑完跳 `/evaluation/formal`(2-3 演示视频小窗口 + per-(codec,crf) 16 行平均表)。

```bash
# speed run(2 段 × 1 codec × 1 crf,少量视频)
PYTHONUTF8=1 uv run python scripts/run_natural_baseline.py \
  --sequences akiyo_cif,bus_cif --codecs x264 --crfs 23

# formal test(全量 6 段 × 3 codec × 4 crf = 72 runs,写 xiph_cif.json)
PYTHONUTF8=1 uv run python scripts/run_natural_baseline.py \
  --codecs x264,x265,vp9
```

`run_*_baseline.py` 支持 `--method`/`--crfs`/`--codecs`/`--frames`/`--sequences`/`--skip-download`;写独立 results 文件(`run_natural`→`results/video/xiph_cif.json`,`run_osu`→`results/video/results.json`),多数据集共存。

**前端**:`/evaluation/run`(EvalRun)顶部 mode 选择器(speed/formal),mode 只影响"数据集子集(--sequences)+ 跳哪个展示页",不在跑代码分叉。`/evaluation/speed`(SpeedResults,视频网格)+ `/evaluation/formal`(FormalResults,平均+演示)。旧 `/evaluation/results`(per-run EvalResults)废弃,重定向到 `/evaluation/formal`。

**后端**:`POST /api/evaluation/run` 接 `dataset_id`/`codecs`/`crfs`/`method`/`sequences`/`mode`,按 dataset 选脚本(xiph_cif→run_natural,osu→run_osu),Popen 传 CLI 参数。

## 4. 结果格式 + 聚合端点

每数据集独立 `results/video/<dataset>.json`:
```json
{ "generated_at": "...", "dataset": "Xiph-CIF-natural", "codecs": [...], "crfs": [...], "git_sha": "...", "runs": [ VideoCompressionResult, ... ] }
```
每 run 携带 `dataset` 字段(envelope dataset 优先,否则文件名 stem:results.json→"default",xiph_cif.json→"xiph_cif" 兜底;但 run_natural 写 envelope "Xiph-CIF-natural")。

**聚合端点**(formal 用):`GET /api/evaluation/results/aggregate?dataset=&method=` → per-(codec,crf) 16 行平均(复用 `aggregate_by_codec_crf`,跨所有 seq 的 PSNR/SSIM/码率/bpp/fps/压缩比/count)。前端 `getAggregatedResults()` 调用。

`/api/evaluation/results`(聚合 `results/video/*.json`,per-run dataset_name)/`/results/compare`(分组)/`/outputs`(按需视频流)。

## 5. 端到端自检

```bash
uv run python -m benchmark.video.verify
# 合成 even(64×64)+ odd(65×63)视频 → canny → x264/vp9@crf23 → 校验指标/产物 → ALL PASS
```

## 6. 增删 codec / 提取器

- 新 codec:`codecs/` 加 `@register_codec("name")` 的 `VideoCodec` 子类,设 `encoder`/`family`/`ext`。基类统一 `-pix_fmt yuv420p` + 奇数尺寸 pad。
- 新提取器:`extractors/` 加 `@register("name")` 子类,实现 `extract(frame_gray)->uint8`。
- 均通过注册表自动发现,无需改 CLI。

## 关键约定

- **统一 `-pix_fmt yuv420p`**:编码统一像素格式(chroma 一致、PSNR 可比),decode 用 `gray` 出单通道 PNG。
- **奇数尺寸**:`encode_args` 自动 pad 到偶,重建裁回原 W×H 再算指标。
- **per-run dataset**:每 run 携带所属数据集名;evaluation 聚合多数据集 results 文件,按 dataset 过滤。
- **两阶段解耦**:阶段1 产物(无损 PNG + manifest)既是阶段2 输入也是质量基准;`--extract-only`/`--skip-extract` 独立运行。
- **评测逻辑统一**:speed/formal 不在跑代码分叉,只通过 `--sequences` 子集 + 展示页区分。

## 常用命令

```bash
uv run python -m benchmark.video.verify                          # 自检
uv run python -m benchmark.video --input datasets/raw/xiph_cif --method canny  # 全流程(目录 glob 多序列)
uv run python scripts/run_natural_baseline.py --codecs x264,x265,vp9           # formal 全量 baseline
uv run python scripts/run_natural_baseline.py --sequences akiyo_cif --codecs x264 --crfs 23  # speed 少量
# 前端:访问 /evaluation/speed(视频网格)+ /evaluation/formal(平均指标)
```
