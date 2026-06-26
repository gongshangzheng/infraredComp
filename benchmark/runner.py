"""Image compression benchmark — runner."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Register HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIC = True
except ImportError:
    HAS_HEIC = False

from .metrics import CompressionResult
from .traditional import (
    compress_jpeg,
    compress_webp,
    compress_webp_lossless,
    compress_png,
    compress_jpeg2000,
    compress_jpeg2000_lossy,
    compress_avif,
    compress_avif_lossless,
    compress_heic,
    JPEG_QUALITIES,
    WEBP_QUALITIES,
    AVIF_QUALITIES,
    HEIC_QUALITIES,
)
from .visualize import generate_report


DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "FLIR_ADAS_1_3" / "FLIR_ADAS_1_3"
# We benchmark on thermal_16_bit images (14-bit IR data in 16-bit TIFF)
THERMAL_SUBDIRS = ["train/thermal_16_bit", "val/thermal_16_bit", "video/thermal_16_bit"]


def load_infrared_images(max_images: int = 50) -> list[tuple[str, np.ndarray]]:
    """Load 16-bit thermal infrared images from the FLIR dataset."""
    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset directory not found: {DATASET_ROOT}")
        print("Run `uv run python scripts/download_dataset.py` first.")
        sys.exit(1)

    images = []
    extensions = {".tif", ".tiff"}

    for subdir in THERMAL_SUBDIRS:
        search_dir = DATASET_ROOT / subdir
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in extensions:
                try:
                    img = Image.open(f)
                    arr = np.array(img)
                    if arr.size == 0:
                        continue
                    images.append((f.stem, arr))
                except Exception as e:
                    print(f"Warning: could not load {f}: {e}")
                if len(images) >= max_images:
                    break
        if len(images) >= max_images:
            break

    if not images:
        print(f"ERROR: No thermal images found under {DATASET_ROOT}")
        sys.exit(1)

    print(f"Loaded {len(images)} thermal 16-bit images")
    print(f"  First image: {images[0][0]} — shape {images[0][1].shape}, dtype {images[0][1].dtype}, "
          f"range [{images[0][1].min()}, {images[0][1].max()}]")
    return images


def build_codec_list(include_learned: bool = False, device: str = "cpu"):
    """Build list of (name, function) codec pairs."""
    codecs = []

    # Traditional
    for q in JPEG_QUALITIES:
        codecs.append((f"JPEG q{q}", lambda img, name, q=q: compress_jpeg(img, name, quality=q)))
    for q in WEBP_QUALITIES:
        codecs.append((f"WebP q{q}", lambda img, name, q=q: compress_webp(img, name, quality=q)))
    codecs.append(("WebP-lossless", compress_webp_lossless))
    codecs.append(("PNG", compress_png))
    codecs.append(("JPEG2000-lossless", compress_jpeg2000))
    codecs.append(("JPEG2000 q75", lambda img, name: compress_jpeg2000_lossy(img, name, quality=75)))

    # AVIF
    for q in AVIF_QUALITIES:
        codecs.append((f"AVIF q{q}", lambda img, name, q=q: compress_avif(img, name, quality=q)))
    codecs.append(("AVIF-lossless", compress_avif_lossless))

    # HEIC (if available)
    if HAS_HEIC:
        for q in HEIC_QUALITIES:
            codecs.append((f"HEIC q{q}", lambda img, name, q=q: compress_heic(img, name, quality=q)))
        print("Included HEIC compression codecs")

    # Learned
    if include_learned:
        try:
            from .learned import LEARNED_MODELS, compress_learned
            for model_name, qualities in LEARNED_MODELS:
                for q in qualities:
                    def make_fn(m=model_name, qual=q):
                        return lambda img, name: compress_learned(img, name, m, qual, device)
                    label = f"{model_name}-q{q}"
                    codecs.append((label, make_fn()))
            print(f"Included {sum(len(qs) for _, qs in LEARNED_MODELS)} learned compression codecs")

            # ELIC (separate model, custom checkpoint loading)
            try:
                from .learned import ELIC_QUALITIES, compress_elic
                for q in ELIC_QUALITIES:
                    def make_elic(qual=q):
                        return lambda img, name: compress_elic(img, name, qual, device)
                    label = f"ELIC-q{q}"
                    codecs.append((label, make_elic()))
                print(f"Included {len(ELIC_QUALITIES)} ELIC compression codecs")
            except Exception as e:
                print(f"Warning: could not load ELIC codecs: {e}")
        except Exception as e:
            print(f"Warning: could not load learned codecs: {e}")

    return codecs


def run_benchmark(
    images: list[tuple[str, np.ndarray]],
    include_learned: bool = False,
    device: str = "cpu",
) -> list[CompressionResult]:
    """Run all codecs on all images and return results."""
    codecs = build_codec_list(include_learned=include_learned, device=device)
    total = len(codecs) * len(images)
    results: list[CompressionResult] = []
    done = 0

    for codec_name, codec_fn in codecs:
        for img_name, img in images:
            try:
                result = codec_fn(img, img_name)
                results.append(result)
            except Exception as e:
                print(f"  ERROR [{codec_name}] on {img_name}: {e}")
            done += 1

        # Progress per codec
        pct = done * 100 // total
        print(f"  [{codec_name}] done ({pct}% total)")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Infrared Image Compression Benchmark")
    parser.add_argument("--max-images", type=int, default=50, help="Max images to test")
    parser.add_argument("--learned", action="store_true", help="Include neural codecs (slow)")
    parser.add_argument("--device", default="cpu", help="Device for learned codecs")
    args = parser.parse_args()

    images = load_infrared_images(max_images=args.max_images)
    results = run_benchmark(images, include_learned=args.learned, device=args.device)

    # Generate visualizations (PNG charts) + CSV/Markdown
    generate_report(results)

    # Generate HTML report
    from .html_report import generate_html_report
    num_images = len(images)
    res = f"{images[0][1].shape[1]}×{images[0][1].shape[0]}"
    generate_html_report(results, num_images=num_images, resolution=res)


if __name__ == "__main__":
    main()
