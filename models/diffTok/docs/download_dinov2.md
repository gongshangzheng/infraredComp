# Download DINOv2 Checkpoints

This guide explains how to download DINOv2 pretrained model checkpoints for REPA loss computation in Poet.

## Overview

DINOv2 (from Meta AI) is used for computing perceptual loss in the REPA (Representation Alignment) loss module. The checkpoints need to be downloaded before training if you plan to use REPA loss.

## Available Models

| Model | Description | Parameters | Embed Dim | Recommended Use |
|-------|-------------|------------|-----------|-----------------|
| `dinov2_vits14` | ViT-S/14 (small) | 21M | 384 | ✅ **Recommended** - Fast & efficient |
| `dinov2_vitb14` | ViT-B/14 (base) | 86M | 768 | Better quality, slower |
| `dinov2_vitl14` | ViT-L/14 (large) | 304M | 1024 | High quality, memory intensive |
| `dinov2_vitg14` | ViT-g/14 (giant) | 1.1B | 1536 | Best quality, requires GPU |

## Quick Start

### 1. List Available Models

```bash
python scripts/download_dinov2.py --list
```

### 2. Download a Model

**Recommended: Download ViT-S/14 (small)**
```bash
python scripts/download_dinov2.py --model dinov2_vits14
```

**Download ViT-B/14 (base)**
```bash
python scripts/download_dinov2.py --model dinov2_vitb14
```

### 3. Verify Download

Test that the checkpoint can be loaded:
```bash
python scripts/download_dinov2.py --model dinov2_vits14 --test checkpoints/dinov2_vits14.pt
```

## Advanced Usage

### Custom Output Directory

```bash
python scripts/download_dinov2.py --model dinov2_vits14 --output path/to/checkpoints
```

### Force Re-download

```bash
python scripts/download_dinov2.py --model dinov2_vits14 --force
```

### All Options

```bash
python scripts/download_dinov2.py --help
```

## Using Downloaded Checkpoints

After downloading, you can use the checkpoints in your training configuration:

**Option 1: Use torch.hub.load (recommended)**
```python
from src.losses.repa_loss import DINOv2

# This will use the cached/downloaded model
dinov2 = DINOv2('dinov2_vits14')
```

**Option 2: Load from local file**
```python
import torch
import torch.hub

# Load model architecture
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)

# Load from local checkpoint
checkpoint = torch.load('checkpoints/dinov2_vits14.pt', map_location='cpu')
model.load_state_dict(checkpoint)
```

## File Structure

After downloading, you'll have:

```
checkpoints/
├── dinov2_vits14.pt              # Model checkpoint
├── dinov2_vits14_metadata.txt    # Metadata (hash, size, etc.)
└── ...
```

## Troubleshooting

### Download Fails

If download fails, try:
1. Check your internet connection
2. Verify model name is correct (use `--list`)
3. Try with `--force` flag to re-download

### Out of Memory

If you get OOM errors during download:
1. Download on CPU first (script uses CPU by default)
2. Use a smaller model (e.g., `dinov2_vits14` instead of `dinov2_vitl14`)

### Permission Errors

Make sure you have write permissions:
```bash
mkdir -p checkpoints
chmod u+w checkpoints
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchhub
- tqdm (for progress bars)

Install dependencies:
```bash
pip install torch torchvision tqdm
```

## References

- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [REPA Loss Documentation](./repa_loss.md)
