#!/usr/bin/env python3
"""
Download ImageNet-1k dataset from Hugging Face.

This script downloads the ImageNet-1k dataset to the project's data directory.
"""

from pathlib import Path
from huggingface_hub import snapshot_download
import os


def main() -> None:
    # Get the script directory and project root
    script_path = Path(__file__).resolve()
    print(f"[Script] Script path: {script_path}")

    script_dir = script_path.parent
    print(f"[Script] Script directory: {script_dir}")

    project_dir = script_dir.parent
    print(f"[Project] Project root: {project_dir}")

    # Set data directory path
    data_dir = project_dir / "data" / "imagenet"
    print(f"[Data] Data directory: {data_dir}")

    # Set Hugging Face token via environment variable HF_TOKEN
    # Get token from: https://huggingface.co/settings/tokens
    hf_token = os.environ.get("HF_TOKEN")

    # Set mirror endpoint for faster download in China
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    print("[DOWNLOAD] Starting ImageNet-1k dataset download...")
    print(f"[Target] Target directory: {data_dir}")

    # Create target directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    snapshot_download(
        repo_id="imagenet-1k",
        repo_type="dataset",
        local_dir=str(data_dir),
        token=hf_token,
        max_workers=4,
    )

    print("[DOWNLOAD] Download complete!")
    print(f"[Dataset] Dataset location: {data_dir}")

    # List downloaded files
    print("\nDownloaded files:")
    for i, item in enumerate(sorted(data_dir.rglob("*"))[:20]):
        if item.is_file():
            print(f"  {item.relative_to(data_dir)}")


if __name__ == "__main__":
    main()
