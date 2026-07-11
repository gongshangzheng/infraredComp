# infraredComp — 红外图像压缩 Benchmark

## 项目目标

在 FLIR ADAS 1.3 热成像数据集上，对比评测各种图像压缩算法的**压缩质量**和**速度**，包括：

1. **传统压缩算法** — JPEG、WebP、PNG、JPEG2000（有损/无损）
2. **神经压缩算法** — 基于 CompressAI 的预训练模型（Balle 2018、Minnen 2018、Cheng 2020 等）

## 评测指标

| 指标 | 说明 |
|------|------|
| PSNR | 峰值信噪比 (dB)，越高越好 |
| SSIM | 结构相似性，越高越好 |
| BPP | 比特每像素 (bits per pixel)，越低越好 |
| Enc/Dec Time | 编解码耗时 (ms) |
| Compression Ratio | 压缩比 |

## 快速开始

```bash
# 安装依赖
uv sync

# 1. 下载数据集 (~15GB)
uv run python scripts/download_dataset.py

# 2. 运行 benchmark（仅传统算法）
uv run python -m benchmark.runner --max-images 50

# 3. 运行 benchmark（含神经压缩，首次需下载模型）
uv run python -m benchmark.runner --max-images 50 --learned
```

## 项目结构

```
infraredComp/
├── benchmark/
│   ├── __init__.py
│   ├── metrics.py       # PSNR, SSIM 等指标
│   ├── traditional.py   # JPEG, WebP, PNG, JPEG2000
│   ├── learned.py       # CompressAI 神经压缩模型
│   └── runner.py        # 主 benchmark 运行器
├── scripts/
│   └── download_dataset.py  # 数据集下载脚本
├── main.py              # 入口点
└── pyproject.toml
```

## 支持的压缩算法

### 传统算法
- **JPEG** — 质量级别: 95, 75, 50, 25
- **WebP** — 有损质量级别: 95, 75, 50, 25 + 无损
- **PNG** — 无损
- **JPEG2000** — 有损 (q75) + 无损

### 神经压缩 (CompressAI)
- **bmshj2018-factorized** — Factorized Prior (Balle 2018)
- **bmshj2018-hyperprior** — Hyperprior (Balle 2018)
- **mbt2018-mean** — Mean Scale Hyperprior (Minnen 2018)
- **mbt2018** — Scale Hyperprior (Minnen 2018)
- **cheng2020-anchor** — Channel Autoregressive (Cheng 2020)

每个模型支持质量级别 1, 4, 8。

## 数据集

- **FLIR ADAS 1.3** — FLIR 热成像数据集，包含约 10k+ 张红外图像
- 来源: [Kaggle](https://www.kaggle.com/datasets/deepnewbie/flir-thermal-images-dataset)
- 下载: `uv run python scripts/download_dataset.py`（需 Kaggle 凭证；`--version N` 锁版本，`--force` 覆盖）
- 位置可配置: `INFRACOMP_DATASETS_DIR` 环境变量重定位数据集树（默认 `<repo>/datasets`），详见 `datasets/README.md`
