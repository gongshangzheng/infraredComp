"""
DINOv2 Checkpoint Download Script

This script downloads DINOv2 pretrained models and saves them to the checkpoints directory.
The models can then be loaded locally without requiring internet access.

Usage:
    python scripts/download_dinov2.py --model dinov2_vits14
    python scripts/download_dinov2.py --model dinov2_vitb14 --output checkpoints/pretrained
    python scripts/download_dinov2.py --list  # List available models
"""
import os
import sys
import argparse
import torch
import hashlib
from pathlib import Path
from tqdm import tqdm


# DINOv2 available models
DINOV2_MODELS = {
    'dinov2_vits14': {
        'description': 'DINOv2 ViT-S/14 (small)',
        'params': '21M',
        'embed_dim': 384,
    },
    'dinov2_vitb14': {
        'description': 'DINOv2 ViT-B/14 (base)',
        'params': '86M',
        'embed_dim': 768,
    },
    'dinov2_vitl14': {
        'description': 'DINOv2 ViT-L/14 (large)',
        'params': '304M',
        'embed_dim': 1024,
    },
    'dinov2_vitg14': {
        'description': 'DINOv2 ViT-g/14 (giant)',
        'params': '1.1B',
        'embed_dim': 1536,
    },
}


def get_file_hash(filepath, algorithm='sha256'):
    """Calculate file hash for verification."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def download_with_progress(url, destination):
    """Download file with progress bar."""
    import urllib.request

    def progress_hook(block_num, block_size, total_size):
        progress.update(block_num * block_size - progress.n)

    progress = tqdm(total=0, unit='B', unit_scale=True, desc=f"Downloading {Path(destination).name}")

    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        progress.close()
        return True
    except Exception as e:
        progress.close()
        print(f"[ERROR] Error downloading {url}: {e}")
        return False


def download_dinov2_checkpoint(model_name, output_dir, force=False):
    """
    Download DINOv2 checkpoint from torch hub.

    Args:
        model_name: Name of the DINOv2 model (e.g., 'dinov2_vits14')
        output_dir: Directory to save the checkpoint
        force: Force re-download even if file exists

    Returns:
        bool: True if successful, False otherwise
    """
    if model_name not in DINOV2_MODELS:
        print(f"[ERROR] Unknown model: {model_name}")
        print(f"Available models: {', '.join(DINOV2_MODELS.keys())}")
        return False

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = DINOV2_MODELS[model_name]
    print(f"📥 Downloading {model_info['description']}")
    print(f"   Parameters: {model_info['params']}")
    print(f"   Embed dim: {model_info['embed_dim']}")

    try:
        # Load model from torch hub (this will download the weights)
        print(f"[Loading] Loading {model_name} from torch hub...")
        device = 'cpu'  # Load on CPU to save GPU memory
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)

        # Save the model
        checkpoint_path = output_dir / f"{model_name}.pt"
        if checkpoint_path.exists() and not force:
            print(f"[WARNING] Checkpoint already exists: {checkpoint_path}")
            response = input("Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("[ERROR] Download cancelled")
                return False

        print(f"[SAVE] Saving checkpoint to {checkpoint_path}...")
        torch.save(model.state_dict(), checkpoint_path)

        # Calculate file hash
        file_hash = get_file_hash(checkpoint_path)
        print(f"[DOWNLOAD] Checkpoint saved successfully!")
        print(f"   Path: {checkpoint_path}")
        print(f"   Size: {checkpoint_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   SHA256: {file_hash}")

        # Create a simple metadata file
        metadata_path = output_dir / f"{model_name}_metadata.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"DINOv2 Model: {model_name}\n")
            f.write(f"Description: {model_info['description']}\n")
            f.write(f"Parameters: {model_info['params']}\n")
            f.write(f"Embedding dimension: {model_info['embed_dim']}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"SHA256: {file_hash}\n")

        print(f"[METADATA] Metadata saved to {metadata_path}")
        return True

    except Exception as e:
        print(f"[ERROR] Error downloading {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_local_checkpoint(model_name, checkpoint_path):
    """
    Test loading a local checkpoint.

    Args:
        model_name: Name of the DINOv2 model
        checkpoint_path: Path to the checkpoint file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"[Test] Testing checkpoint loading from {checkpoint_path}...")
        device = 'cpu'

        # Load model architecture
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)

        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        print("[DOWNLOAD] Checkpoint loaded successfully!")
        print(f"   Model: {model_name}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return True

    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download DINOv2 checkpoints for Poet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download ViT-S/14 model to default location
  python scripts/download_dinov2.py --model dinov2_vits14

  # Download ViT-B/14 model to custom location
  python scripts/download_dinov2.py --model dinov2_vitb14 --output checkpoints/pretrained

  # List all available models
  python scripts/download_dinov2.py --list

  # Force re-download even if file exists
  python scripts/download_dinov2.py --model dinov2_vitl14 --force

  # Test loading an existing checkpoint
  python scripts/download_dinov2.py --model dinov2_vits14 --test checkpoints/dinov2_vits14.pt
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        help='DINOv2 model name (e.g., dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='checkpoints',
        help='Output directory for checkpoints (default: checkpoints)'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if checkpoint exists'
    )

    parser.add_argument(
        '--test', '-t',
        type=str,
        metavar='CHECKPOINT_PATH',
        help='Test loading a checkpoint from the specified path'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available DINOv2 models'
    )

    args = parser.parse_args()

    # List models
    if args.list:
        print("[List] Available DINOv2 models:")
        print("=" * 60)
        for model_name, info in DINOV2_MODELS.items():
            print(f"\n{model_name}:")
            print(f"  Description: {info['description']}")
            print(f"  Parameters: {info['params']}")
            print(f"  Embed dim: {info['embed_dim']}")
        print("=" * 60)
        return 0

    # Test loading checkpoint
    if args.test:
        if not args.model:
            print("[ERROR] Error: --model is required when using --test")
            return 1
        success = load_local_checkpoint(args.model, args.test)
        return 0 if success else 1

    # Download checkpoint
    if not args.model:
        parser.print_help()
        print("\n[ERROR] Error: --model is required (use --list to see available models)")
        return 1

    success = download_dinov2_checkpoint(args.model, args.output, args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
