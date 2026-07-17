---
name: difftok
description: |
  infraredComp 中 DiffTok / ContourVQVAE（TiTok 风格 1D VQ tokenizer）的训练、评测与使用指南。
  覆盖模型架构、BSDS500 数据准备、训练命令、checkpoint 加载、stage-2 codec 注册、
  前端 /evaluation 集成和常见问题。
  触发场景：(1) 训练/微调 ContourVQVAE (2) 用 difftok codec 跑视频压缩评测
  (3) 接入新 VQ 提取器/codec (4) 调试 BCE+VQ loss 或 dead code
---

# DiffTok / ContourVQVAE 使用指南

DiffTok 是 infraredComp 内置的**灰度轮廓图像 1D VQ tokenizer**，基于 TiTok 思想：用少量可学习 latent tokens 从整图抽取紧凑离散表示，再通过 decoder 重建。它作为 stage-2 视频 codec（`difftok`）注册在 `benchmark/video/codecs/` 中，每帧独立编码为 `num_latent` 个整数 token ids。

## 1. 模型架构

```text
输入 x [B,1,H,W]
  ↓
ContourEncoder (ViT, patch + learnable latent tokens)
  → latent_tokens [B, num_latent, enc_dim]
  ↓
Linear(enc_dim → token_dim)
  ↓
VectorQuantizer1d (EMA, dead-code reset, kmeans++ 可选初始化)
  → z_q [B, num_latent, token_dim] + token ids [B, num_latent]
  ↓
Linear(token_dim → dec_dim)
  ↓
ContourDecoder (mask tokens + latent tokens → unpatchify)
  → logits [B,1,H,W]
  ↓ sigmoid
重建图像 [B,1,H,W]
```

关键设计：
- **无 AdaLN/conditioning**：`ContourBlock` 是纯 pre-norm ViT block，`ContourFinalLayer` 也无调制，仅保留 Attention + Mlp + RMSNorm。
- **1D latent tokens**：默认 64 个 token，每个 token 指向大小为 1024 的 codebook 中的一项，每帧固定 `64 × log2(1024) = 640 bits`。
- **BCE + VQ**：训练用 `binary_cross_entropy_with_logits` 加 `pos_weight=10`，配合 VQ commitment/codebook loss。

核心文件：

| 文件 | 作用 |
|------|------|
| `models/diffTok/src/nets/contour_vqae.py` | 顶层 `ContourVQAE`：forward / encode_indices / decode_indices |
| `models/diffTok/src/encoders/contour_encoder.py` | 可学习 latent token 的 ViT encoder |
| `models/diffTok/src/decoders/contour_decoder.py` | 从量化 token 重建图像的 ViT decoder |
| `models/diffTok/src/components/contour_block.py` | 无 AdaLN 的 pre-norm ViT block（内联 Mlp，无 timm 依赖） |
| `models/diffTok/src/components/contour_final_layer.py` | 无 AdaLN 的输出层 |
| `models/diffTok/src/quantizers/quantizer1d.py` | EMA VQ，含 dead code reset、usage stats、perplexity |
| `models/diffTok/src/losses/rd_loss.py` | `rd_loss_difftok`：BCE + VQ loss |

## 2. 依赖

DiffTok 需要 `omegaconf` 读配置、`einops` 做 rearrange，已经在项目依赖中；若缺失：

```bash
uv pip install omegaconf einops
```

模型代码刻意**不依赖 timm**（`contour_block.py` 内联了 `Mlp`）。

## 3. 数据准备

DiffTok 训练推荐用 **BSDS500 GT 软边缘图**（二值边缘标注平均后的灰度图）。也可用 `imagenet-<split>` 或自行提取的 contour PNG。

### 3.1 BSDS500

把 BSDS500 数据集放到 `datasets/BSDS500`（可用 junction/symlink 指向大容量盘）：

```text
datasets/BSDS500/
├── images/{train,val,test}/*.jpg
└── groundTruth/{train,val,test}/*.mat
```

转成 `datasets/contour/bsds_<split>_gt/frame_*.png`：

```bash
# 全部
uv run python scripts/convert_bsds_gt.py

# 只转 val
uv run python scripts/convert_bsds_gt.py --splits val

# 指定尺寸（训练用 128）
uv run python scripts/convert_bsds_gt.py --splits train,val --save-size 128
```

产物：
- `datasets/contour/bsds_train_gt/frame_*.png`
- `datasets/contour/bsds_val_gt/frame_*.png`
- `datasets/contour/bsds_test_gt/frame_*.png`
- 每个目录下 `manifest.json`

### 3.2 ImageNet contour（可选）

`scripts/imagenet_contour_dataset.py` 提供在线流式提取 `ImageNetContourDataset`，以及预提取 PNG 读取 `ContourPNGDataset`。需要 `datasets/imagenet/data/*.parquet`。

## 4. 训练

### 4.1 配置文件

默认配置：`configs/difftok/bsds_contour.yaml`

```yaml
model:
  image_size: 128
  patch_size: 8
  inout_chans: 1
  num_latent: 64
  mlp_ratio: 4.0
  qkv_bias: true
  drop: 0.0
  attn_drop: 0.0
  qk_norm: false
  encoder: { dim: 384, depth: 6, num_heads: 6 }
  decoder: { dim: 384, depth: 6, num_heads: 6 }

quantizer:
  token_dim: 32
  codebook_size: 1024
  commitment_cost: 0.25
  reservoir_size: 4096
  dead_code_threshold: 1.0
  decay: 0.99
  eps: 1.0e-5
  use_l2_norm: false

train:
  lr: 1.0e-4
  batch_size: 32
  bce_pos_weight: 10.0
  epochs: 500
  save_interval: 50
```

### 4.2 启动训练

```bash
uv run python scripts/train_model.py \
  --model difftok \
  --dataset bsds-train \
  --size 128 \
  --batch 32 \
  --lr 1e-4 \
  --epochs 500 \
  --run-id difftok_bsds_001 \
  --device cuda
```

参数说明：
- `--model difftok`：走 `build_model()` 的 difftok 分支，从 `configs/difftok/bsds_contour.yaml` 实例化 `ContourVQAE`。
- `--dataset bsds-train`：`resolve_training_dataset()` 解析到 `datasets/contour/bsds_train_gt/`。
- `--size 128`：必须和配置 `model.image_size` 一致。
- `--batch 32`：配置里也有 `train.batch_size`，命令行优先。
- `--lr 1e-4`：AdamW 学习率。

训练产物：
- `results/training/checkpoints/{run_id}.pth`：最终 checkpoint（state_dict）。
- `results/training/checkpoints/{run_id}.best.pth`：eval PSNR 最高时的 checkpoint。
- `results/training/metrics.json`：训练曲线、状态、可视化路径。
- `results/training/logs/{run_id}.log`：日志。
- `results/training/viz/{run_id}/`：每 epoch 重建可视化。

### 4.3 从 checkpoint 续训 / warm-start

```bash
# 加载权重，新 run_id，fresh optimizer（不继承曲线）
uv run python scripts/train_model.py \
  --model difftok --dataset bsds-train \
  --load difftok_bsds_001 \
  --run-id difftok_bsds_002 --epochs 100

# 继承上一条 run 的曲线，从 last+1 续跑
uv run python scripts/train_model.py \
  --model difftok --dataset bsds-train \
  --resume difftok_bsds_001 \
  --run-id difftok_bsds_002 --epochs 100
```

`--load` 只加载权重；`--resume` 继承 `loss_series/test_metrics/viz`。

### 4.4 Loss 说明

`models/diffTok/src/losses/rd_loss.py`：

```python
def rd_loss_difftok(logits, x0, vq_loss, pos_weight=10.0):
    pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, x0, pos_weight=pw)
    loss = bce + vq_loss
    ...
    return loss, loss.item(), psnr, 0.0
```

- `pos_weight=10`：边缘像素稀疏（通常 5-15%），加大边缘像素的 BCE 权重。
- `vq_loss`：EMA 模式下只含 commitment_loss（codebook 完全由 EMA 驱动，不走梯度）。
- 返回的 `bpp` 固定为 0.0（占位），因为 difftok 的 bpp 由 token 数量决定，不是熵模型 likelihood。

训练日志示例：

```text
[difftok_bsds_001] [train] epoch 1 (1/500) loss=0.5234 psnr=12.34 bpp=0.000
```

### 4.5 训练时 eval

`--eval-every N` 默认对图像模型开启。eval 会在 held-out `bsds-val`（或 `--eval-split`）上算 PSNR/BPP/loss，并保存最佳 checkpoint。

```bash
uv run python scripts/train_model.py \
  --model difftok --dataset bsds-train \
  --eval-split val --eval-every 5 \
  --run-id difftok_bsds_001 --epochs 500
```

## 5. 评测（作为 stage-2 codec）

训练好的 checkpoint 可通过 `--checkpoint` 传给评测脚本。

### 5.1 BSDS val 单图伪序列评测

```bash
# formal 全量
uv run python scripts/run_bsds_baseline.py \
  --mode formal --codecs difftok --checkpoint results/training/checkpoints/difftok_bsds_001.best.pth

# speed 子集
uv run python scripts/run_bsds_baseline.py \
  --mode speed --max-images 20 --codecs difftok --checkpoint results/training/checkpoints/difftok_bsds_001.best.pth
```

结果写入：
- `results/video/bsds_val.json`
- `results/video/bsds_val_speed.json`

### 5.2 自然视频轮廓序列评测

先跑 stage-1 提取 contour.mp4，再跑 stage-2：

```bash
# 提取
uv run python -m benchmark.video \
  --input datasets/raw/xiph_cif/akiyo_cif.y4m --method canny --extract-only

# 评测（仅 difftok）
uv run python -m benchmark.video \
  --input datasets/contour/akiyo_cif/canny \
  --skip-extract --codecs difftok --crfs 1 \
  --checkpoint results/training/checkpoints/difftok_bsds_001.best.pth
```

`--crfs 1` 对 difftok 仅作为占位；实际质量不可调（固定 codebook + num_latent），`qualities=[1]`。

### 5.3 与传统 codec 对比

```bash
uv run python scripts/run_bsds_baseline.py \
  --mode formal \
  --codecs difftok,x264,x265,vp9 \
  --crfs 18,23,28,33 \
  --checkpoint results/training/checkpoints/difftok_bsds_001.best.pth
```

## 6. 代码集成点

### 6.1 训练侧

`scripts/train_model.py`：
- 顶部 `from models.diffTok.src.losses.rd_loss import rd_loss_difftok`
- `build_model()` 中 `model_id == "difftok"` 分支
- 训练循环中 `if args.model == "difftok"` 分支
- `resolve_training_dataset()` 中 `bsds-<split>` 路由，自动传 `out_channels=1`

`scripts/imagenet_contour_dataset.py`：
- `ContourPNGDataset` 支持 `out_channels=1`，difftok 训练时只取首通道。

### 6.2 评测侧

`benchmark/video/codecs/difftok.py`：
- 注册为 `difftok`
- `encode_inprocess`：逐帧 `encode_indices` → uint16 token ids → pickle
- `decode_inprocess`：token ids → `decode_indices` → 重建帧
- `_estimated_bytes` 按 `num_latent * log2(codebook_size)` 估算，无熵编码

`benchmark/video/codecs/__init__.py`：
- `from . import difftok as _difftok`
- `_NON_IMG_QUALITIES["difftok"] = [1]`
- `_CODEC_META["difftok"]`

`benchmark/video/stage2_benchmark.py`：
- `run_benchmark()` 接受 `checkpoint_path` 参数，透传给 neural codec。

### 6.3 后端 / 前端

`server/routers/evaluation.py`：
- `_bsds_datasets()` 提供 `bsds-val` 数据集入口
- `/run` 收到 `dataset_id == "bsds-val"` 时调用 `scripts/run_bsds_baseline.py`

前端 `/evaluation/run` 的 mode 选择器 + dataset 下拉可看到 `bsds-val`。

## 7. 常用命令速查

```bash
# 数据准备
uv run python scripts/convert_bsds_gt.py --splits train,val --save-size 128

# 训练
uv run python scripts/train_model.py --model difftok --dataset bsds-train \
  --size 128 --batch 32 --lr 1e-4 --epochs 500 --run-id difftok_bsds_001 --device cuda

# 评测（BSDS val）
uv run python scripts/run_bsds_baseline.py --mode formal --codecs difftok \
  --checkpoint results/training/checkpoints/difftok_bsds_001.best.pth

# 评测（视频序列）
uv run python -m benchmark.video --input datasets/contour/akiyo_cif/canny \
  --skip-extract --codecs difftok --crfs 1 \
  --checkpoint results/training/checkpoints/difftok_bsds_001.best.pth

# 导入测试
uv run python -c "
from models.diffTok.src.nets.contour_vqae import ContourVQAE
from omegaconf import OmegaConf
import torch
cfg = OmegaConf.load('configs/difftok/bsds_contour.yaml')
m = ContourVQAE(cfg).eval()
x = torch.zeros(2, 1, 128, 128)
out, vq = m(x)
print(out.shape, vq['quantizer_loss'])
"
```

## 8. 常见问题

**Q: 训练报 `No module named 'omegaconf'` 或 `einops'`？**
A: `uv pip install omegaconf einops`

**Q: `convert_bsds_gt.py` 报 BSDS500 源目录不存在？**
A: 需先建 `datasets/BSDS500`（可用 symlink/junction 到数据盘）。

**Q: difftok 的 BPP 为什么和传统 codec 不直接可比？**
A: difftok 目前无熵编码，按 `num_latent * log2(codebook_size)` 估算理论 bits；真实码流是 pickle 包，比理论值大。RD 曲线更多展示“固定码率下的重建质量”。

**Q: 想改 num_latent / codebook_size？**
A: 改 `configs/difftok/bsds_contour.yaml`，然后重新训练。注意：已训练的 checkpoint 与新的配置维度必须匹配，否则 `load_state_dict` 会失败。

**Q: codebook usage 很低 / dead code 多？**
A: `VectorQuantizer1d` 已内置 dead code reset（`dead_code_threshold`）。若仍低，可尝试：
- 减小 `commitment_cost`
- 增大 batch size
- 初始化 codebook：`model.quantizer.initialize_codebook(samples)`（训练前用一批数据 kmeans++ 初始化）

**Q: 能否把 difftok 当 stage-1 提取器用？**
A: 当前 difftok 是压缩 codec（stage-2），不是边缘提取器。它的输入已经是轮廓图，输出是重建轮廓图。

## 9. 关键文件清单

| 文件 | 说明 |
|------|------|
| `configs/difftok/bsds_contour.yaml` | 默认模型/训练配置 |
| `models/diffTok/src/nets/contour_vqae.py` | 顶层模型 |
| `models/diffTok/src/encoders/contour_encoder.py` | Encoder |
| `models/diffTok/src/decoders/contour_decoder.py` | Decoder |
| `models/diffTok/src/quantizers/quantizer1d.py` | EMA VQ |
| `models/diffTok/src/losses/rd_loss.py` | BCE+VQ loss |
| `models/diffTok/src/components/contour_block.py` | 无 AdaLN block |
| `models/diffTok/src/components/contour_final_layer.py` | 无 AdaLN 输出层 |
| `scripts/train_model.py` | 训练入口 |
| `scripts/convert_bsds_gt.py` | BSDS500 GT 转换 |
| `scripts/run_bsds_baseline.py` | BSDS val 评测 |
| `scripts/imagenet_contour_dataset.py` | ImageNet / PNG Dataset |
| `benchmark/video/codecs/difftok.py` | stage-2 codec 封装 |
| `benchmark/video/codecs/__init__.py` | codec 注册表 |
| `server/routers/evaluation.py` | 后端 /evaluation 路由 |
