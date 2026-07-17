"""
Convert old single-file checkpoint to new 4-file format.

Usage:
    python scripts/convert_checkpoint.py --checkpoint checkpoints/stage1_mask_16384/latest.pt
"""
import argparse
import os
import json
import torch


def convert_checkpoint(old_checkpoint_path: str, output_dir: str = None):
    """Convert old single-file checkpoint to new 4-file format."""
    print(f"[CONVERT] Loading: {old_checkpoint_path}")

    # Load old checkpoint
    ckpt = torch.load(old_checkpoint_path, map_location='cpu')
    global_step = ckpt['global_step']

    # Load loss from .json file
    checkpoint_dir = os.path.dirname(old_checkpoint_path)
    checkpoint_name = os.path.basename(old_checkpoint_path).replace('.pt', '.json')
    json_path = os.path.join(checkpoint_dir, checkpoint_name)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        loss = json_data['loss']

    print(f"  global_step: {global_step}, loss: {loss:.6f}")

    # Determine output directory
    if output_dir is None:
        parent_dir = os.path.dirname(old_checkpoint_path)
        output_dir = os.path.join(parent_dir, 'latest')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save model
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save({'global_step': global_step, 'model_state_dict': ckpt['model_state_dict']}, model_path)
    size_gb = os.path.getsize(model_path) / (1024**3)
    print(f"  [CONVERT] Saved model.pt ({size_gb:.2f}GB)")

    # 2. Save EMA1
    ema1_path = os.path.join(output_dir, 'ema1.pt')
    torch.save({'global_step': global_step, 'ema_params1': ckpt['ema_state_dict']['ema_params1']}, ema1_path)
    size_gb = os.path.getsize(ema1_path) / (1024**3)
    print(f"  [CONVERT] Saved ema1.pt ({size_gb:.2f}GB)")

    # 3. Save EMA2
    ema2_path = os.path.join(output_dir, 'ema2.pt')
    torch.save({'global_step': global_step, 'ema_params2': ckpt['ema_state_dict']['ema_params2']}, ema2_path)
    size_gb = os.path.getsize(ema2_path) / (1024**3)
    print(f"  [CONVERT] Saved ema2.pt ({size_gb:.2f}GB)")

    # 4. Save optimizers
    optimizers_data = {
        'global_step': global_step,
        'optimizer_state_dicts': ckpt['optimizer_state_dicts'],
        'lr_scheduler_state_dicts': ckpt['lr_scheduler_state_dicts']
    }
    optimizers_path = os.path.join(output_dir, 'optimizers.pt')
    torch.save(optimizers_data, optimizers_path)
    size_gb = os.path.getsize(optimizers_path) / (1024**3)
    print(f"  [CONVERT] Saved optimizers.pt ({size_gb:.2f}GB)")

    # 5. Save metadata
    metadata = {'global_step': global_step, 'loss': float(loss)}
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [CONVERT] Saved metadata.json")

    print(f"\n[CONVERT] Complete: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert old checkpoint to new format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to old checkpoint file')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    convert_checkpoint(args.checkpoint, args.output)
