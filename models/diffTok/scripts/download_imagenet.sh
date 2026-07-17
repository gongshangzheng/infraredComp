#!/bin/bash
# 文件名: ~/code/diffTok/scripts/download_imagenet.sh

# 获取当前脚本的绝对路径
SCRIPT_PATH="$(readlink -f "$0")"
echo "🔍 脚本路径: $SCRIPT_PATH"

# 获取脚本所在的目录（scripts 文件夹）
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
echo "📁 脚本目录: $SCRIPT_DIR"

# 获取项目根目录（脚本目录的父目录）
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "📂 项目根目录: $PROJECT_DIR"

# 设置数据集路径（在项目根目录下的 data/imagenet-1k）
DATA_DIR="$PROJECT_DIR/data/imagenet-1k"
echo "💾 数据目录: $DATA_DIR"

# 设置你的 Hugging Face token（从 https://huggingface.co/settings/tokens 获取）
# export HF_TOKEN="your_token_here"  # 从 https://huggingface.co/settings/tokens 获取，或提前设好环境变量

# 设置镜像站加速（国内用户推荐）
export HF_ENDPOINT=https://hf-mirror.com

echo "🚀 开始下载 ImageNet-1k 数据集..."
echo "📁 目标目录: $DATA_DIR"

# 创建目标目录
mkdir -p "$DATA_DIR"

# 下载数据集
hf download --repo-type dataset imagenet-1k \
  --token "$HF_TOKEN" \
  --local-dir "$DATA_DIR" \
  --max-workers 4

echo "✅ 下载完成！"
echo "📂 数据集位置: $DATA_DIR"
ls -la "$DATA_DIR" | head -20
