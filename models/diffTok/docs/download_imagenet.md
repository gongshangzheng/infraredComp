# ImageNet Dataset Download Guide

This guide explains how to download and prepare the ImageNet dataset for training Poet.

## Overview

The `scripts/download_imagenet.py` script helps you:
1. Download ImageNet dataset (with instructions)
2. Extract the tar files
3. Convert to WebDataset format for efficient training

## Prerequisites

### 1. ImageNet Access

ImageNet requires registration. Visit https://www.image-net.org/download.php to:
- Create an account
- Request access to ILSVRC2012
- Download the dataset files

### 2. Disk Space

You'll need approximately **300GB** of free disk space:
- ~150GB for downloaded tar files
- ~150GB for temporary extracted files
- ~150GB for final WebDataset shards

Use the `--cleanup` flag to automatically remove temporary files after conversion.

### 3. Install Dependencies

```bash
pip install webdataset pillow tqdm
```

## Quick Start

### Step 1: Get Download Instructions

```bash
python scripts/download_imagenet.py --download-only
```

This will show you exactly which files to download.

### Step 2: Download ImageNet (Manual)

Download these files from the ImageNet website:
- **ILSVRC2012_img_train.tar** (~138GB) - Training images
- **ILSVRC2012_img_val.tar** (~6.3GB) - Validation images

Place both files in: `./data/imagenet/`

### Step 3: Convert to WebDataset

```bash
python scripts/download_imagenet.py --output_dir ./data/imagenet --convert
```

This will:
1. Extract the tar files
2. Convert images to WebDataset format
3. Create optimized shards for training

## Advanced Usage

### Process Specific Splits

**Training data only** (faster, less disk space):
```bash
python scripts/download_imagenet.py --split train
```

**Validation data only**:
```bash
python scripts/download_imagenet.py --split val
```

**Both (default)**:
```bash
python scripts/download_imagenet.py --split both
```

### Customize Shard Size

Larger shards = fewer files but more memory usage:

```bash
# 500MB per shard (more files, less memory)
python scripts/download_imagenet.py --shard-size 0.5

# 2GB per shard (fewer files, more memory)
python scripts/download_imagenet.py --shard-size 2
```

### Automatic Cleanup

Remove temporary extracted files after creating shards:

```bash
python scripts/download_imagenet.py --cleanup
```

This saves ~150GB of disk space but keeps the original tar files.

### Full Example

```bash
python scripts/download_imagenet.py \
    --output_dir ./data/imagenet \
    --split both \
    --shard-size 1 \
    --cleanup
```

## Output Structure

After conversion, your directory structure will be:

```
./data/imagenet/
├── ILSVRC2012_img_train.tar      # Original download (keep these)
├── ILSVRC2012_img_val.tar        # Original download (keep these)
├── train_shards/                 # Training WebDataset shards
│   ├── train-0000.tar
│   ├── train-0001.tar
│   └── ... (approximately 138 files)
└── val_shards/                   # Validation WebDataset shards
    ├── val-0000.tar
    ├── val-0001.tar
    └── ... (approximately 7 files)
```

## Using in Training

Update your training configuration to use the downloaded data:

```yaml
# configs/stage1/your_config.yaml
dataset:
  name: imagenet
  train_shards_path: ./data/imagenet/train_shards/train-*.tar
  eval_shards_path: ./data/imagenet/val_shards/val-*.tar
  num_train_examples: 1281167  # ImageNet training set size
```

Then train as usual:

```bash
python scripts/train_stage1.py -c configs/stage1/your_config.yaml
```

## Troubleshooting

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'webdataset'`

**Solution**:
```bash
pip install webdataset pillow tqdm
```

### Insufficient Disk Space

**Error**: `OSError: [Errno 28] No space left on device`

**Solutions**:
1. Use a different output directory with more space:
   ```bash
   python scripts/download_imagenet.py --output_dir /path/to/large/drive
   ```
2. Use `--cleanup` to remove temporary files:
   ```bash
   python scripts/download_imagenet.py --cleanup
   ```
3. Process only training data first:
   ```bash
   python scripts/download_imagenet.py --split train --cleanup
   # Then process validation separately
   python scripts/download_imagenet.py --split val --cleanup
   ```

### Memory Issues During Conversion

**Error**: `MemoryError` or process killed during conversion

**Solution**: Reduce shard size to use less memory:
```bash
python scripts/download_imagenet.py --shard-size 0.5  # 500MB shards
```

### Corrupted Images

**Warning**: Some images may fail to verify

**Solution**: The script automatically skips corrupted images and continues processing. This is normal for ImageNet (a few corrupted images exist in the official dataset).

### Validation Class Mapping

**Note**: The validation data uses a simplified class mapping. For accurate evaluation, you need the official `ILSVRC2012_validation_ground_truth.txt` file.

Place it at:
```
./data/imagenet/validation_ground_truth.txt
```

## Performance Tips

1. **Use SSD storage**: WebDataset conversion is I/O intensive
2. **Parallel processing**: The script automatically uses multiple cores
3. **Network storage**: If using network storage, increase shard size to reduce file count
4. **Monitor progress**: The script shows progress bars for all operations

## References

- ImageNet: https://www.image-net.org/
- WebDataset: https://github.com/webdataset/webdataset
- Poet Training: See main README
