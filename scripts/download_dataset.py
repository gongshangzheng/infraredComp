"""Download the FLIR ADAS 1.3 thermal dataset via KaggleHub.

Usage:
    uv run python scripts/download_dataset.py                 # latest version
    uv run python scripts/download_dataset.py --version 3     # pin a specific version
    uv run python scripts/download_dataset.py --force         # overwrite existing dir

Requires Kaggle credentials:
  - Option A: export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>
  - Option B: place a kaggle.json at ~/.kaggle/kaggle.json
    (https://www.kaggle.com/settings/account → Create New Token)

The datasets tree location defaults to <repo>/datasets but can be relocated via
the INFRACOMP_DATASETS_DIR environment variable (large datasets may live outside
the repo). On success a record is appended to datasets/manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Datasets 树位置可经环境变量配置;默认 <repo>/datasets
DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(PROJECT_ROOT / "datasets")))

DEFAULT_SLUG = "deepnewbie/flir-thermal-images-dataset"
DATASET_NAME = "FLIR_ADAS_1_3"
TARGET_DIR = DATASETS_DIR / DATASET_NAME
MANIFEST_PATH = DATASETS_DIR / "manifest.json"


def _check_credentials() -> None:
    """Fail fast with a clear message if Kaggle credentials are missing."""
    has_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not has_env and not kaggle_json.exists():
        print(
            "ERROR: Kaggle credentials not found.\n"
            "  Option A: export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>\n"
            "  Option B: place a kaggle.json at ~/.kaggle/kaggle.json\n"
            "  Get a token at: https://www.kaggle.com/settings/account → Create New Token",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_handle(slug: str, version: int | None) -> str:
    """kagglehub dataset handle; appending /versions/N pins a specific version."""
    return f"{slug}/versions/{version}" if version else slug


def _parse_version(download_path: str) -> int | None:
    """Best-effort extract version from kagglehub's cache path (.../versions/N/...)."""
    parts = Path(download_path).parts
    for i, p in enumerate(parts):
        if p == "versions" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _flatten(download_path: str) -> None:
    """Move downloaded contents up to TARGET_DIR and remove nested cache dirs."""
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    shutil.move(download_path, TARGET_DIR)

    # Remove the nested kagglehub cache tree (datasets/datasets/...) if created
    nested_cache = DATASETS_DIR / "datasets"
    if nested_cache.is_dir():
        shutil.rmtree(nested_cache)


def _write_manifest(slug: str, version: int | None) -> None:
    """Refresh the record for this dataset in datasets/manifest.json (list, one entry per name)."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    records: list = []
    if MANIFEST_PATH.exists():
        try:
            loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                records = loaded
        except (json.JSONDecodeError, OSError):
            records = []
    records = [r for r in records if r.get("name") != DATASET_NAME]
    records.append({
        "name": DATASET_NAME,
        "source": f"kaggle:{slug}",
        "version": version,
        "path": str(TARGET_DIR),
        "size_bytes": _dir_size_bytes(TARGET_DIR),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    })
    MANIFEST_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[manifest] wrote {MANIFEST_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the FLIR ADAS 1.3 dataset via KaggleHub."
    )
    parser.add_argument("--slug", default=DEFAULT_SLUG,
                        help=f"Kaggle dataset slug (default: {DEFAULT_SLUG})")
    parser.add_argument("--version", type=int, default=None,
                        help="pin a specific Kaggle dataset version")
    parser.add_argument("--force", action="store_true",
                        help="overwrite TARGET_DIR if it already exists")
    args = parser.parse_args()

    _check_credentials()

    if TARGET_DIR.exists() and not args.force:
        print(f"Target already exists: {TARGET_DIR}")
        print("Pass --force to overwrite (existing data will be deleted).")
        return 0

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    # Cache downloads inside the local datasets folder
    os.environ["KAGGLEHUB_CACHE"] = str(DATASETS_DIR)

    handle = _build_handle(args.slug, args.version)
    print(f"Downloading {handle} ...")
    download_path = kagglehub.dataset_download(handle, force_download=args.force)
    print(f"Downloaded to cache: {download_path}")

    # Parse version before _flatten moves (and invalidates) the cache path
    version = args.version or _parse_version(download_path)
    _flatten(download_path)
    _write_manifest(args.slug, version)

    print(f"Dataset ready at: {TARGET_DIR}")
    if version is not None:
        print(f"  version: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
