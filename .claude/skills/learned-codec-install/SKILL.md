---
name: learned-codec-install
description: |
  往 infraredComp 安装/集成一个新的学习式压缩模型库的流程 meta-skill。覆盖：装库、加 codec、接 eval+training、下 checkpoint、**为该库建专属 usage skill**（规则：每个学习式压缩库都配一个 skill）。也适用于 CompressAI(已装)/DCVC-RT(待装)/未来其它。
  触发场景：(1) 集成新的学习式压缩库 (2) 加学习式视频 codec (3) 接训练/评测 (4) 给某学习库建 usage skill (5) 不知道某步该改哪
---

# 安装学习式压缩模型库的流程（infraredComp）

infraredComp 的视频 benchmark 默认 codec 走 ffmpeg（x264/x265/svtav1/vp9）。**学习式压缩库**（CompressAI、DCVC-RT、未来其它）走 in-process 神经 codec 路径。集成一个新库的完整流程：

## 规则（写在最前）

> **每个学习式压缩模型库都配一个专属 usage skill**（`.claude/skills/<lib>-usage/SKILL.md`），说明该库的 API（模型/压缩/解压/训练/checkpoint）。不要只在一个总 skill 里塞所有库——每库独立，便于维护 + cherry-pick + 触发。
>
> 已有：`compressai-usage`（CompressAI：image + ssf2020 video）。集成 DCVC-RT 时建 `dcvc-rt-usage`。集成新库 X 时建 `x-usage`。

## 集成流程（6 步）

### 1. 装库
- pip 包 → `pyproject.toml` 加依赖 + `uv sync`。
- 非 pip（如 DCVC-RT 多为 repo）→ vendor 进 `benchmark/video/codecs/external/<lib>/`（clone 子目录或 git submodule）+ 在 `benchmark/video/codecs/<lib>.py` 里 import。
- 记 license + torch/依赖版本兼容性（学习式视频 codec 常依赖特定 torch/cuda op）。

### 2. 加 codec 模块
`benchmark/video/codecs/<lib>.py`，继承 `VideoCodec`（`benchmark/video/codecs/base.py`）：
```python
from .base import VideoCodec, register_codec
@register_codec("<lib-name>")
class XCodec(VideoCodec):
    name = "<lib-name>"; family = "learned-video"; ext = "bin"; is_neural = True
    def __init__(self, crf, preset=None, checkpoint_path=None):
        super().__init__(crf, preset); self.checkpoint_path = checkpoint_path
        # lazy load model (见 step 3 的 _load 模式)
    def encode_inprocess(self, frames, fps) -> bytes: ...   # frames=list[np.ndarray HxW uint8]
    def decode_inprocess(self, bs, n, hw) -> list: ...       # -> list[np.ndarray HxW uint8]
```
- 复用 `benchmark/learned.py` 的 `_img_to_tensor/_tensor_to_img/_pad_to_multiple/_unpad`（min-max norm + gray→3ch + ÷64 pad）。
- bitstream 序列化：`pickle.dumps({"strings":..., "shapes":..., "n":..., "stats":..., "pads":..., "hw":...})`；decode 镜像。
- 在 `benchmark/video/codecs/__init__.py` 加 `from . import <lib> as _<lib>` 注册。
- `crf` 复用为该库的 quality 级（ssf2020: 1-9；DCVC-RT 各自）。

### 3. checkpoint 加载（pretrained + trained override）
`_load_model(quality, device, checkpoint_path=None)`：
- `checkpoint_path` 给 → fresh model + `torch.load(checkpoint_path, weights_only=False)` + `load_state_dict`。
- else → pretrained（库的 pretrained 下载到 `~/.cache/torch/hub/checkpoints/` 或该库指定目录）。
- 必 `model.eval()` + `model.update()`（CompressAI 类熵模型需要）。
- 缓存（`_CACHE` keyed by quality+device+checkpoint）避免每次 build_codec 重载。

### 4. 接 eval + training
- **eval**：`server/routers/evaluation.py /models` 加该 codec 项（`kind: 'learned-video'` + `checkpoint` 字段 = `_trained_checkpoints_for(<lib>)` 扫 trained .pth）；`POST /run` 解析 checkpoint_id（已有分支扩展到 learned-video）。`EvalRun.vue` 选 learned-video codec 时出现 checkpoint+quality 选择器。
- **training**：`scripts/train_model.py` 加视频训练路径分支（`--model <lib-name>`）——`VideoFrameSequenceDataset`（轮廓连续帧 list[Tensor]）+ 该库的 forward/loss + warm-start（从 pretrained checkpoint 起）。image 模型维持旧 per-frame 路径。

### 5. 下 pretrained checkpoint
`scripts/download_learned_checkpoints.py` 加该库的 pretrained 拉取（库的 model_urls/release/GDrive/HF）→ 缓存目录。幂等 + `--force`。
**效果检验由用户自行跑**（不自动）：下完 checkpoint → `uv run python -m benchmark.video --codecs <lib-name> ...`（pretrained）→ 看轮廓基线 PSNR/bpp（学习式在轮廓 OOD，预期偏低）→ 决定是否 fine-tune。

### 6. 建该库的 usage skill（强制）
`.claude/skills/<lib>-usage/SKILL.md`：该库的模型清单、compress/decompress API、训练 loss、checkpoint 格式、本库映射（`benchmark/video/codecs/<lib>.py` + `scripts/train_model.py` 视频路径 + download 脚本）。**这是规则（见上）**。

## 验证（集成后）
1. codec 注册：`uv run python -c "from benchmark.video.codecs import list_codecs; print(list_codecs())"` 含 `<lib-name>`。
2. pretrained 评测（用户跑）：`uv run python -m benchmark.video --input datasets/raw/osu_color_thermal/seq1.mp4 --method canny --codecs <lib-name> --crfs 1,5,9` → results.json 一条 learned-video run（in-process encode/decode 分支生效，bitstream.bin + recon PNG + PSNR/SSIM/bpp）。
3. 训练：`POST /api/training/run {model_id:<lib-name>, dataset:contour/…, epochs:2}` → metrics.json loss_series + trained .pth。
4. checkpoint→eval：trained .pth 自动出现在 `/api/evaluation/models` 的 `<lib-name>` 项 `checkpoint` 字段 → EvalRun 选它评测。

## 关键约定
- **in-process 神经 codec 不走 ffmpeg**：`is_neural=True` → `stage2_benchmark.benchmark_codec` 走 `encode_inprocess/decode_inprocess` 分支（写 bytes 到 bitstream 文件做 size 统计 + 写 recon PNG 让共享 metrics 管线工作）。legacy ffmpeg codec 不受影响。
- **轮廓数据 OOD**：学习式库 pretrained 在自然图像/视频训练，在**轮廓帧**上效果差 → 大概率要 fine-tune（training 视频路径，warm-start from pretrained）。
- **序列化**：bitstream = pickle(strings+shapes+stats+pads+hw)；decode 读回。size = len(bytes) → bpp/压缩比照常。
- **每库一 skill**：`<lib>-usage` skill 独立建，不混。
