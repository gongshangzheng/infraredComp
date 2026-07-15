---
name: integrate-third-party-model
description: 往 infraredComp 接入一个第三方模型（边缘检测器/压缩 codec 等）的通用流程 meta-skill。覆盖：判定它是 stage1 提取器还是 stage2 codec、权重 vendor+gitignore、代码 submodule-vs-vendor 决策、@register 注册、web 暴露、GPU/torch 环境、验证清单。适用于 CompressAI/DCVC/HED/PiDiNet/未来任意第三方模型。
---

# 接入第三方模型 — 通用流程

把一个第三方（GitHub）模型接入 infraredComp，分两类定位 + 一条代码组织原则。

## 1. 先判定：提取器 还是 codec？

| 维度 | stage1 提取器（轮廓提取） | stage2 codec（压缩） |
|---|---|---|
| 输入/输出 | 原始灰度帧 → 边缘图(uint8 HxW) | 轮廓帧序列 → 码流 + 重建帧 |
| 注册 | `benchmark/video/extractors/` + `@register("name")`（`extractors/base.py` 的 `ContourExtractor.extract(frame_gray)->uint8`） | `benchmark/video/codecs/` + `@register_codec("name")`（`codecs/base.py` 的 `VideoCodec`） |
| 自动发现 | import `extractors/__init__.py` 即进 `/methods` 端点 + Formal/Speed/EvalRun 下拉 | import `codecs/__init__.py` 即进 `/codecs` 端点 + catalog |
| 范例 | canny/sobel/hed/**pidinet** | x264/mpeg4/ssf2020/dcvc_rt/img-* |

→ 先问「这个模型是产生边缘图，还是产生压缩码流？」定类别，再走对应注册路径。一个模型也可能两个都做（看它的 forward）。

## 2. 权重处理（通用）

- **永远 vendor 到 `third_party/<model>/`**，不要放 `~/.cache`（自包含、可复现、不依赖环境变量）。HED 在 `third_party/hed/`、DCVC checkpoint 在 `third_party/DCVC/checkpoints/`、PiDiNet 在 `third_party/pidinet/`。
- **大二进制权重（>~10MB）必 gitignore** + 写 `scripts/download_<model>_weights.py`（NO_PROXY、幂等、**输出路径与 extractor/codec 默认一致**——吸取 HED 脚本/默认名不一致的教训）。
- **小权重（<~10MB）可二选一**：直接 git 跟踪（最省事，上游 commit 即得）或 gitignore+下载脚本（与大体积一致）。看上游权重在哪：commit 在仓库内（PiDiNet）→ 下载脚本=浅克隆+拷贝；OneDrive/GDrive（DCVC/HED）→ 直链下载脚本。
- **权重永远不 submodule**（上游无关、体积大、且 submodule 不便放单文件）。

## 3. 代码组织：submodule vs vendor（关键原则）

> **只用不改 → submodule 可行**；**会改内部 → vendor 代码直接 git 跟踪，不用 submodule**。

- **submodule**（`git submodule add`，pin 上游 commit）：干净跟踪上游版本、不撑大本仓库。**前提：不改其内部**。范例：`third_party/DCVC`（目前用作 DCVC-RT 推理，不改内部）。clone 本仓库需 `git submodule update --init --recursive`。
- **vendor**（拷源码进 `third_party/<model>/` 直接 git 跟踪）：改内部就是本仓库的正常 commit（可 review/复现/在历史里）。**前提：会改内部，或小模型想省 submodule init**。代价：丢上游自动链接（更新需手动同步）。范例：PiDiNet（vendored `models/` + gitignore 权重，保留改内部自由）、HED（只 vendored prototxt，权重 gitignore）。

**为何改内部就不用 submodule**：submodule 内部修改是它**自己工作树里的 dirty 改动**，没法干净提交到**本**仓库（它有自己的 .git），`git submodule status` 一直显示脏，协作者 clone 后 `submodule update --init` 拉到的是干净上游、看不到你的改动。vendor 后修改=普通 commit，无此问题。

vendor 时记得 `rm -rf third_party/<model>/.git`（去掉 vendor 来源的 git，变成纯 tracked 文件），否则嵌套 git 仓库。

## 4. 注册 + web 暴露

- **自动发现面（无需改）**：`extractors/__init__.py`（或 `codecs/__init__.py`）加 `from . import <name> as _<name>` → `@register`/`@register_codec` 触发 → `/methods`（或 `/codecs`）端点 + Formal/Speed/EvalRun 下拉 registry 驱动。`benchmark/video/__main__.py`、`stage1_extract.py`、`run_all_subprocess.py` 都用 `list_extractors()`/`list_codecs()`，注册名即合法 `--method`/`--codecs` 值。
- **硬编码面（必须手动补）**：
  - `server/routers/evaluation.py` 的 `GET /configs`（约 649 行）`methods` 列表 + 描述串。
  - 前端硬编码 method 选择器：`web/src/views/training/TrainRun.vue`（约 110）、`web/src/views/evaluation/DatasetDetail.vue`（约 140，易漏 hed）。
  - codec 的 qualities：`codecs/__init__.py` 的 `_NON_IMG_QUALITIES` + `_CODEC_META`（仅 codec 需要）。

## 5. GPU/torch 环境

- benchmark 的 learned 部分（torch codec/extractor）跑在 **conda env `compression`**（`C:/Users/wo/.conda/envs/compression/python.exe`，torch 2.11.0+cu130, RTX 5090）。server 本身跑 `uv run`（CPU torch），benchmark 子进程经 `INFRACOMP_BENCH_PYTHON` 走 conda compression。
- stage1（提取器）in-process，继承跑 stage1 的 python——直接用 conda compression python 跑 `python -m benchmark.video --method <name> ...`，extractor 自动拿到 GPU。
- **torch 加载惯法**（仿 `codecs/ssf2020.py:46`、`codecs/learned_image.py`）：`device = "cuda" if torch.cuda.is_available() else "cpu"`、`torch.load(path, map_location=device, weights_only=False)`、`load_state_dict`、`.to(device).eval()`、推理包 `torch.no_grad()`、**模块级 `_CACHE` dict 按 device 缓存**避免每次 `build_extractor`/`build_codec` 重载。

## 6. 验证清单（接入后必跑）

1. `python -c "from benchmark.video.extractors import list_extractors; print(list_extractors())"`（或 `list_codecs()`）含新名。
2. 单帧冒烟：`build_extractor('<name>').extract(gray)` → uint8 HxW，range 0-255，非全黑/全亮。
3. 全流程（提取器）：`python -m benchmark.video --method <name> --input datasets/raw/xiph_cif/akiyo_cif.y4m --extract-only` → `datasets/contour/akiyo_cif/<name>/contour.mp4` + manifest。
4. 重启后端 → `curl http://localhost:8091/api/evaluation/methods`（或 `/codecs`）含新名；前端下拉出现。
5. 跑一组 `<name> × 传统 codec` 的 formal 子集，确认 RD 点与既有方法不同（canny/sobel/hed 不应相同）。

## 7. 范例

| 模型 | 类别 | 代码组织 | 权重 | torch? | 备注 |
|---|---|---|---|---|---|
| HED (s9xie/hed) | 提取器 | 只 vendored `deploy.prototxt`（tracked） | `hed_pretrained_bsds.caffemodel` gitignore | 否（cv2.dnn） | 自定义 Crop 层；末层 sigmoid-fuse，不二次 sigmoid |
| DCVC-RT (microsoft/DCVC) | codec | **submodule**（不改内部） | CVPR2025 pth 在 submodule `checkpoints/`（gitignore） | 是 | rans C++ ext 必须 build；DMCI+DMC+DPB seeding |
| PiDiNet (hellozhuo/pidinet) | 提取器 | **vendor 代码**（models/ tracked） | `table5_pidinet.pth` gitignore（上游 commit 内） | 是 | converted 重参数化；strip DataParallel `module.` 前缀；forward 已 sigmoid |
| YOLOE-26 (ultralytics) | 提取器 | **pip install ultralytics**（不改内部） | `yoloe-26s-seg-pf.pt` gitignore（ultralytics assets 下） | 是 | prompt-free seg（无需类列表）→ mask 边界当 contour；**需彩色输入**；分割→轮廓的语义不同于稠密边缘 |

> **彩色化 pipeline（重要）**：stage1 解码+传 **彩色 BGR** 帧给 `extract(frame)`，各提取器自己决定灰/彩——HED/PiDiNet/YOLOE 训于彩色 BSDS/COCO，吃彩色更对（灰度→3ch 会丢色彩、YOLOE 甚至 0 检出）；canny/sobel 内部 `cvtColor(BGR2GRAY)`。灰度源（红外）解码成 3 同通道，无损。
>
> **分割→轮廓（YOLOE 类）**：分割模型不出边缘图，取 instance mask 边界（`cv2.findContours` + `drawContours`）当 contour——对象轮廓语义，比稠密边缘稀疏，是不同维度的"轮廓"。
>
> **逐帧 vs 原生视频**：当前 stage1 固定"demux 拆帧 + 逐帧 extract"。未来若有原生处理视频的模型，应让提取器自己负责拆帧（base 加 `extract_video` 默认逐帧、可 override 走原生），不强制逐帧。

## 关键文件

- `benchmark/video/extractors/base.py`（ContourExtractor ABC + `@register` + EXTRACTOR_REGISTRY）
- `benchmark/video/extractors/__init__.py`（import 触发注册）
- `benchmark/video/codecs/base.py`（VideoCodec + `@register_codec`）、`codecs/__init__.py`（catalog + qualities）
- `benchmark/video/extractors/hed.py`（cv2.dnn 提取器模板）、`extractors/pidinet.py`（torch 提取器模板）
- `benchmark/video/codecs/ssf2020.py`、`learned_image.py`（torch 加载惯法）
- `benchmark/video/stage1_extract.py`（`extract_contour_video`，method 校验 + idempotent 复用）
- `server/routers/evaluation.py`（`/methods` ~316、`/configs` ~641 硬编码 methods ~649）
- `web/src/views/training/TrainRun.vue`、`web/src/views/evaluation/DatasetDetail.vue`（硬编码 method 选择器）
- `scripts/download_hed_weights.py`、`scripts/download_pidinet_weights.py`（下载脚本模板）
- `.gitignore`（`third_party/hed/*.caffemodel`、`third_party/pidinet/*.pth`）
