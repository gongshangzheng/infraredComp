import os
import shutil

import kagglehub

# Resolve project root (parent of this script's directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# Cache downloads inside the local datasets folder
os.environ["KAGGLEHUB_CACHE"] = DATASETS_DIR

DATASET_SLUG = "deepnewbie/flir-thermal-images-dataset"
TARGET_DIR = os.path.join(DATASETS_DIR, "FLIR_ADAS_1_3")


def _flatten(download_path: str) -> None:
    """Move downloaded contents up to datasets/FLIR_ADAS_1_3 and remove nested cache dirs."""
    if os.path.isdir(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    shutil.move(download_path, TARGET_DIR)

    # Remove the nested kagglehub cache tree (datasets/datasets/...)
    nested_cache = os.path.join(DATASETS_DIR, "datasets")
    if os.path.isdir(nested_cache):
        shutil.rmtree(nested_cache)


# Download latest version (lands in a deeply nested cache folder)
path = kagglehub.dataset_download(DATASET_SLUG)

# Flatten the layout so the data sits directly under datasets/FLIR_ADAS_1_3
_flatten(path)

print("Dataset ready at:", TARGET_DIR)
