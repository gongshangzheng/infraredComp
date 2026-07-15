---
name: contour-video-evaluation
description: |
  Contour-Video 轮廓视频压缩评测库(benchmark/video/)使用指南。管线产物是**无损 contour.mp4**(libx264 -qp 0 yuv420p),**不保留 PNG 帧**;阶段2 从视频解码出临时帧跑评测。两模式评测(speed run 视频网格 / formal test 平均指标),**默认不截断帧**(speed 靠 --sequences 子集加速)。
  触发场景:(1) 从原始视频提取轮廓视频(阶段1,产出 contour.mp4) (2) 跑视频 codec 压缩评测(阶段2,从视频起) (3) speed run(少量视频,视频网格看主观,不截断帧) (4) formal test(全量,per-(codec,crf) 平均指标,不截断帧) (5) 跑 verify 端到端自检 (6) 查看/调整 results 结果格式 / aggregate 端点 (7) 重跑评测(删旧 results JSON 强制刷新)
---

# Contour-Video 轮廓视频压缩评测库

`benchmark/video/` 库 + 评测前端两模式的使用指南。库位于仓库根 `benchmark/video/`。所有路径相对仓库根;先 `cd` 到含 `pyproject.toml` 的仓库根再运行命令。

## 项目结构

```text
infraredComp/benchmark/video/
├── config.py            # 路径常量(datasets/raw|contour, results/video, bitstreams/recon/source/contour_mp4)
├── ffmpeg_util.py       # ffmpeg/ffprobe 发现(INFRACOMP_FFMPEG_BIN → PATH → static_ffmpeg fallback)
├── data.py              # ContourArtifact(带 video_path)/ VideoCompressionResult(含 dataset 字段)
├── extractors/          # 阶段1 可插拔提取器(canny/sobel,@register)
├── stage1_extract.py    # extract_contour_video(产 contour.mp4,删 PNG,temp+swap 原子替换)
│                        #   load_contour_frames(读 PNG)/ load_contour_video_frames(解码 video→np)
├── codecs/              # 阶段2 视频 codec(x264/x265/svtav1/vp9 + ssf2020/dcvc_rt/img-*)
├── stage2_benchmark.py  # benchmark_codec/ run_benchmark(从 video 解码临时帧)/ save_results_json
├── metrics.py           # PSNR/SSIM/时序一致性
├── aggregate.py         # per-(codec,crf) 平均 / RD 曲线 / bests
├── repro.py             # build_metadata(git_sha/codecs/crfs/dataset envelope + frame_cap)
├── visualize.py / html_report.py
├── artifact_io.py       # load_artifact(从 manifest 读 video_path,不再 glob PNG)
├── __main__.py          # CLI(--input 可重复 + 目录 glob,多序列累积)
└── verify.py            # 端到端自检(合成小视频)
```

## 安装

```bash
cd <repo-root>
uv sync                                   # 安装依赖(含 torch/compressai/opencv)
# ffmpeg:无需系统安装,`uv add static-ffmpeg` 内置 ffmpeg+ffprobe(static_ffmpeg fallback);
#         或设 INFRACOMP_FFMPEG_BIN 指向 ffmpeg.exe。static-ffmpeg win32 不含 libsvtav1。
# 网络:本机 conda 代理可能阻塞 pypi 镜像,设 NO_PROXY=* 绕过(tuna 镜像不可达时)。
#       也可直接用 .venv/Scripts/python.exe(Windows)绕过 uv 的网络解析。
```

数据目录:`datasets/raw/`(原始视频)、`datasets/contour/<source>/<method>/`(阶段1 产物,按方法分目录,只含 `contour.mp4` + `manifest.json`)、`results/video/`(阶段2 产物 + 按需展示 mp4 缓存)。`datasets/` 大数据 + `results/video/` 运行产物均不入 git(见 `.gitignore`)。

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

> ⚠️ 下载脚本目前**不写 raw manifest 的 `frame_count`**(只填 fps/size_bytes)。运行后若数据集页序列头显示 "0 帧",需用 ffprobe 数 y4m 的 `nb_read_packets` 回填:
> ```bash
> FP=.venv/Lib/site-packages/static_ffmpeg/bin/win32/ffprobe.exe
> # conda compression python(无需 uv 网络):
> "$USERPROFILE/.conda/envs/compression/python.exe" -c "
> import json, subprocess
> from pathlib import Path
> FP='$FP'
> mf=Path('datasets/raw/xiph_cif/manifest.json')
> m=json.loads(mf.read_text(encoding='utf-8'))
> for s in m['sequences']:
>     n=subprocess.run([FP,'-v','error','-select_streams','v:0','-count_packets',
>         '-show_entries','stream=nb_read_packets','-of','csv=p=0',s['file']],
>         capture_output=True,text=True).stdout.strip()
>     s['frame_count']=int(n) if n.isdigit() else 0
> mf.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
> "
> ```
> 真实帧数:akiyo 300 / bus 150 / city 300 / flower 250 / foreman 300 / mobile 300(5–10s @30fps)。

## 1. 阶段1 提取轮廓视频

```bash
# 从 y4m 提取 canny 轮廓视频(全帧,默认不截断)
uv run python -m benchmark.video --input datasets/raw/xiph_cif/akiyo_cif.y4m --method canny --extract-only

# 整个 xiph_cif 目录(自动 glob *.y4m,多序列累积)
uv run python -m benchmark.video --input datasets/raw/xiph_cif --method canny --extract-only
```

**产物**:`datasets/contour/<source>/<method>/contour.mp4` + `manifest.json`。**不产出 PNG 帧**。

管线细节:
- `extract_contour_video(raw_input, method, frames=None)`:默认 `frames=None` 不截断。
- temp+swap 原子替换:先在 `out_dir.tmp/` 重建 → 成功后 `rmtree out_dir` + `rename(.tmp → out_dir)`。失败/中断不毁既有产物。
- 内部流程:demux raw → 灰度 PNG(`_raw_frames`,完成后清掉)→ canny 写 `frame_*.png`(临时)→ **`libx264 -qp 0 -pix_fmt yuv420p` 拼无损 `contour.mp4`**(奇数尺寸 pad 到偶)→ **删 `frame_*.png`**。
- `manifest.json` 字段:`source_name` / `method` / `frame_count` / `fps` / `width` / `height` / `frames_dir` / `video_path` / `duration_s`。
- `skip_if_exists=True`(训练流幂等):manifest 有 `video_path` 且文件在 → 复用,跳过重建。
- 返回的 `ContourArtifact`:`video_path` 指向 contour.mp4;`frame_paths=[]`(PNG 已删,不再持有);`frames_dir` = out_dir。

新增提取器:`extractors/` 加 `@register("name")` 的 `ContourExtractor` 子类,实现 `extract(frame_gray)->uint8`。

## 2. 阶段2 压缩评测

```bash
# 全流程:提取 + 压缩评测(--input 可重复/目录 glob,多序列累积到一个 results.json)
uv run python -m benchmark.video --input datasets/raw/xiph_cif \
  --method canny --crfs 18,23,28,33 --codecs x264,x265,svtav1,vp9

# 仅阶段2(复用 contour.mp4 产物;--input 指向 contour 目录而非 raw)
uv run python -m benchmark.video --input datasets/contour/akiyo_cif/canny --skip-extract \
  --crfs 23,28 --codecs x264,vp9
```

codec:`x264`(h264)、`x265`(hevc)、`svtav1`(av1,**static-ffmpeg win32 缺 libsvtav1,跑会失败**)、`vp9`。CRF 0-63(越大质量越低/码率越低);**默认全帧不截断**。

### stage2 从视频起,不直接读 PNG

`run_benchmark(artifact, ...)` 起首:
1. 若 `artifact.video_path` 存在 → `ffmpeg -i video -vf crop={w}:{h}:0:0 -pix_fmt gray -vsync 0 temp/frame_%06d.png` 把**无损 contour.mp4 解码到临时帧目录**(crop 回原尺寸,奇数尺寸还原)。
2. 构造临时 artifact 视图(`frames_dir=tmp`、`frame_paths=glob(tmp)`,其余字段同原 artifact),传入 `benchmark_codec`。
3. 跑完 `rmtree temp`(临时 PNG 不落盘持久)。
4. 若无 `video_path`(旧目录)→ 维持原 PNG 路径。

`benchmark_codec` 自身不变,继续用 `frames_dir` / `load_contour_frames(artifact)` / `encode_args(frames_dir, fps, bitstream)` 读临时 PNG。neural codec(ssf2020 / dcvc_rt / img-*)用 `load_contour_frames` 拿 frame_list。

直接读 `artifact.video_path` 拿 np 数组(不经临时 PNG):`load_contour_video_frames(artifact) -> (N,H,W) uint8`,ffmpeg rawvideo pipe + crop 掉 pad。

## 3. 评测两模式(speed run / formal test)

**评测逻辑统一**(一套 stage1+stage2),差异只在数据集子集 + 展示页。**两模式默认都不截断帧**(speed 靠 `--sequences` 子集加速,formal 全量全帧)。

| 模式 | 目的 | CLI | 展示页 |
|------|------|-----|--------|
| speed run | 少量视频,视频网格看主观 | `--sequences <stem1,stem2>`(seq 子集)+ 少量 codec/crf | `/evaluation/speed`(视频按 codec 分排,每格 `<video preload=none>` 默认黑屏点击加载,filter 缩范围) |
| formal test | 全量,平均指标 | 不传 `--sequences`(全量 seq)+ 全 codec/crf | `/evaluation/formal`(2-3 演示视频小窗口 + per-(codec,crf) 16 行平均表) |

```bash
# speed run(2 段 × 1 codec × 1 crf,少量视频;**全帧**)
PYTHONUTF8=1 uv run python scripts/run_natural_baseline.py \
  --sequences akiyo_cif,bus_cif --codecs x264 --crfs 23

# formal test(全量 6 段 × 3 codec × 4 crf = 72 runs,写 xiph_cif.json;**全帧**)
PYTHONUTF8=1 uv run python scripts/run_natural_baseline.py \
  --codecs x264,x265,vp9
```

`run_*_baseline.py` 支持 `--method` / `--crfs` / `--codecs` / `--frames` / `--sequences` / `--skip-download`。**`--frames` 默认 `None` 不截断**;若显式传 `--frames N` 才限帧。写独立 results 文件(`run_natural` → `results/video/xiph_cif.json`,`run_osu` → `results/video/results.json`),多数据集共存。

⚠️ **重跑评测**:`load_existing()` 按 `seq|codec|crf` key 跳过已存在 run。若改了管线(无截断 / 改了 codec)想**全量刷新**指标,必须**先删旧 results JSON**(`rm results/video/xiph_cif.json`),否则相同 key 会被标 SKIP、跑不到新管线。

**前端**:`/evaluation/run`(EvalRun)顶部 mode 选择器(speed/formal),mode 只影响"数据集子集(--sequences)+ 跳哪个展示页",不在跑代码分叉。`/evaluation/speed`(SpeedResults,视频网格)+ `/evaluation/formal`(FormalResults,平均+演示)。旧 `/evaluation/results`(per-run EvalResults)废弃,重定向到 `/evaluation/formal`。

**后端**:`POST /api/evaluation/run` 接 `dataset_id` / `codecs` / `crfs` / `method` / `sequences` / `mode`,按 dataset 选脚本(xiph_cif → `run_all_subprocess.py`,osu → `run_osu_baseline.py`),Popen 传 CLI 参数。**`frames` 已固定为 `None`**,从不传 `--frames`;speed/formal 都全帧。

## 4. 结果格式 + 聚合端点

每数据集独立 `results/video/<dataset>.json`:
```json
{ "generated_at": "...", "dataset": "Xiph-CIF-natural", "codecs": [...], "crfs": [...], "git_sha": "...", "frame_cap": null, "runs": [ VideoCompressionResult, ... ] }
```
每 run 携带 `dataset` 字段(envelope dataset 优先;results.json→"default",xiph_cif.json→"xiph_cif" 兜底;但 `run_natural` 写 envelope "Xiph-CIF-natural")。

**聚合端点**(formal 用):`GET /api/evaluation/results/aggregate?dataset=&method=` → per-(codec,crf) 16 行平均(复用 `aggregate_by_codec_crf`)。前端 `getAggregatedResults()` 调用。

`/api/evaluation/results`(聚合 `results/video/*.json`)/`/results/compare`(分组)/`/outputs`(按需视频流)。

## 5. 端到端自检

```bash
uv run python -m benchmark.video.verify
# 合成 even(64×64)+ odd(65×63)视频 → canny → x264/vp9@crf23 → 校验指标/产物 → ALL PASS
# 产物:仅 contour.mp4 + manifest(无 PNG);verify 会写合成 results.json 到 results/video/results.json,跑完可删。
```

## 6. 增删 codec / 提取器

- 新 codec:`codecs/` 加 `@register_codec("name")` 的 `VideoCodec` 子类,设 `encoder` / `family` / `ext`。基类统一 `-pix_fmt yuv420p` + 奇数尺寸 pad。
- 新提取器:`extractors/` 加 `@register("name")` 子类,实现 `extract(frame_gray)->uint8`。
- 均通过注册表自动发现,无需改 CLI。

## 关键约定

- **管线产物是无损 `contour.mp4`**(`libx264 -qp 0 -pix_fmt yuv420p`,Y 通道精确还原),**不保留 PNG 帧**;页面展示由后端从 contour.mp4 转有损 H.264(crf18 yuv420p)缓存于 `results/video/contour_mp4/`,经 `/outputs/` 服务。
- **stage2 临时帧**:`run_benchmark` 把 contour.mp4 解码到临时 PNG 目录(crop 回原尺寸),跑完 `rmtree`;持久 contour 目录**永远不存 PNG**。
- **无截断为默认**:`frames=None` 是 stage1 + run_natural + run_all_subprocess + 后端 /run 的默认值。speed run 加速靠 `--sequences` 子集,不靠限帧。
- **奇数尺寸**:lossless stitch pad 到偶;stage2 临时帧 ffmpeg `-vf crop={w}:{h}:0:0` 还原;`load_contour_video_frames` 同样 crop 掉 pad。
- **统一 `-pix_fmt yuv420p`**:编码统一像素格式(chroma 一致、PSNR 可比),decode 用 `gray` 出单通道 PNG。
- **per-run dataset**:每 run 携带所属数据集名;evaluation 聚合多数据集 results 文件,按 dataset 过滤。
- **两阶段解耦**:阶段1 产物(无损 contour.mp4 + manifest)是阶段2 输入与质量基准;`--extract-only` / `--skip-extract` 独立运行。
- **评测逻辑统一**:speed/formal 不在跑代码分叉,只通过 `--sequences` 子集 + 展示页区分。
- **结果 JSON 不可增量跨管线刷新**:`load_existing` 按 key 跳过已存在 run;改管线后重跑前**先删旧 results JSON**。

## 数据集页面展示管线

数据集详情页(`/evaluation/datasets/{id}`)序列展开:左=原始视频、右=轮廓视频(多方法用 n-tabs),都是可播放 mp4。媒体来源:
- 原始视频(浏览器可播):`source/{seq}.mp4` 经 `/outputs/source/{seq}.mp4`(由 `_ensure_source_video` 从 raw 截到 contour 帧窗口生成)。
- 轮廓视频:`contour_mp4/{seq}_{method}.mp4` 经 `/outputs/contour_mp4/{seq}_{method}.mp4`(由 `_ensure_contour_video` 从 contour.mp4 转有损;**只读 contour.mp4,无 PNG 回退**)。
- 原始数据 `seq.file` 字段(给 `/datasets/{id}/media/{path}`):从 raw manifest 解析,后端 `_load_raw_datasets` 剥 `datasets/` 前缀后返回相对 DATASETS_DIR 的路径(如 `raw/xiph_cif/akiyo_cif.y4m`)。

## 常用命令

```bash
# 阶段1
uv run python -m benchmark.video --input datasets/raw/xiph_cif --method canny  # 全流程(目录 glob 多序列)
uv run python -m benchmark.video --input datasets/raw/xiph_cif/akiyo_cif.y4m --method canny --frames 30  # 显式限帧(默认不限)

# 阶段2 / 评测
uv run python -m benchmark.video.verify                                                # 自检
uv run python scripts/run_natural_baseline.py --codecs x264,x265,vp9                   # formal 全量 baseline(全帧)
uv run python scripts/run_natural_baseline.py --sequences akiyo_cif --codecs x264 --crfs 23  # speed 少量(全帧)
# 重跑前: rm results/video/xiph_cif.json  (load_existing 按 key 跳过,会跑不到)

# 前端
# /evaluation/datasets/{id}    序列展开,左右两栏视频(原始 / 轮廓)
# /evaluation/speed             视频网格(speed run 展示)
# /evaluation/formal            平均指标(formal 展示)
```
