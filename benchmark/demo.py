"""Generate a visual demo comparing compression artifacts across codecs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .traditional import (
    compress_avif,
    compress_jpeg,
    compress_png,
    compress_webp,
)

# Representative codecs shown in the demo. PNG acts as the lossless reference.
_DEMO_CODECS = [
    ("JPEG q75", lambda img, name: compress_jpeg(img, name, quality=75)),
    ("WebP q75", lambda img, name: compress_webp(img, name, quality=75)),
    ("AVIF q75", lambda img, name: compress_avif(img, name, quality=75)),
    ("bmshj-factorized-q1", lambda img, name: _learned(img, name, "bmshj2018-factorized", 1)),
    ("mbt-mean-q4", lambda img, name: _learned(img, name, "mbt2018-mean", 4)),
    ("PNG", compress_png),
]


def _learned(img: np.ndarray, name: str, model: str, quality: int):
    """Helper to call learned codec with default CPU device."""
    from .learned import compress_learned

    return compress_learned(img, name, model, quality, device="cpu")


def _find_demo_image() -> tuple[str, np.ndarray]:
    """Load the first available 16-bit thermal image from the FLIR dataset."""
    dataset_root = (
        Path(__file__).resolve().parent.parent
        / "datasets"
        / "FLIR_ADAS_1_3"
        / "FLIR_ADAS_1_3"
    )
    subdirs = ["train/thermal_16_bit", "val/thermal_16_bit", "video/thermal_16_bit"]
    for subdir in subdirs:
        search_dir = dataset_root / subdir
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.iterdir()):
            if f.suffix.lower() in {".tif", ".tiff"}:
                arr = np.array(Image.open(f))
                if arr.size:
                    return f.name, arr
    raise FileNotFoundError("No thermal 16-bit image found for demo")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize any numeric image to uint8 for display."""
    if img.dtype == np.uint8:
        return img
    img_min, img_max = img.min(), img.max()
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)


def _crop_center(img: np.ndarray, size: int) -> np.ndarray:
    """Crop the center ``size x size`` region."""
    h, w = img.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top : top + size, left : left + size]


def _error_map(original: np.ndarray, decoded: np.ndarray) -> np.ndarray:
    """Compute an absolute error map scaled to uint8."""
    if original.shape != decoded.shape:
        decoded = _match_shape(decoded, original.shape)
    if original.dtype != decoded.dtype:
        if decoded.dtype == np.uint8 and original.dtype == np.uint16:
            decoded = decoded.astype(np.uint16) * 256
        elif original.dtype == np.uint8 and decoded.dtype == np.uint16:
            original = original.astype(np.uint16) * 256
    diff = np.abs(original.astype(np.float32) - decoded.astype(np.float32))
    max_err = diff.max()
    if max_err > 0:
        diff = (diff / max_err * 255).astype(np.uint8)
    return diff


def _match_shape(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Simple center crop/pad to match target (H, W) for error computation."""
    h, w = target_shape[:2]
    if img.shape[:2] == (h, w):
        return img
    # If dimensions differ due to codec quirks, resize via PIL.
    pil_img = Image.fromarray(_to_uint8(img))
    pil_img = pil_img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.array(pil_img)
    if len(target_shape) == 3 and arr.ndim == 2:
        arr = np.stack([arr] * target_shape[2], axis=-1)
    return arr


def generate_demo_figure(
    output_dir: Path,
    crop_size: int = 256,
) -> tuple[Path, str]:
    """
    Generate a comparison figure and return (path, image_name).

    The figure shows a center crop of the original image, reconstructions from
    JPEG/WebP/AVIF, their error heatmaps, and a PNG lossless reference.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_name, original = _find_demo_image()
    original_uint8 = _to_uint8(original)
    original_crop = _crop_center(original_uint8, crop_size)

    results = []
    for label, fn in _DEMO_CODECS:
        results.append(fn(original, image_name))

    n_cols = len(results)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 8))

    # Top row: original + reconstructions for all lossy codecs.
    axes[0, 0].imshow(original_crop, cmap="gray")
    axes[0, 0].set_title(f"Original ({original.shape[1]}×{original.shape[0]})", fontsize=11)
    axes[0, 0].axis("off")

    lossy_results = results[:-1]  # everything except PNG reference
    for idx, result in enumerate(lossy_results):
        decoded = result.decoded if result.decoded is not None else np.zeros_like(original)
        decoded_crop = _crop_center(_to_uint8(decoded), crop_size)
        axes[0, idx + 1].imshow(decoded_crop, cmap="gray")
        axes[0, idx + 1].set_title(
            f"{result.codec}\nPSNR {result.psnr:.1f} dB  BPP {result.bpp:.2f}",
            fontsize=9,
        )
        axes[0, idx + 1].axis("off")

    # Bottom row: error maps for all lossy codecs + PNG reference.
    for idx, result in enumerate(lossy_results):
        decoded = result.decoded if result.decoded is not None else np.zeros_like(original)
        err = _error_map(original_crop, _crop_center(_to_uint8(decoded), crop_size))
        axes[1, idx].imshow(err, cmap="hot")
        axes[1, idx].set_title(f"{result.codec} Error", fontsize=9)
        axes[1, idx].axis("off")

    png_result = results[-1]
    png_decoded = png_result.decoded if png_result.decoded is not None else np.zeros_like(original)
    png_crop = _crop_center(_to_uint8(png_decoded), crop_size)
    axes[1, -1].imshow(png_crop, cmap="gray")
    axes[1, -1].set_title(
        f"PNG (lossless)\nBPP {png_result.bpp:.2f}",
        fontsize=9,
    )
    axes[1, -1].axis("off")

    fig.suptitle(
        f"Infrared Compression Effect Demo — Center {crop_size}×{crop_size} crop of {image_name}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_path = output_dir / "demo_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Demo figure saved: {out_path}")
    return out_path, image_name


def demo_html_section(image_name: str, crop_size: int = 256) -> str:
    """Return the HTML section string for the demo figure."""
    return f"""
  <!-- DEMO -->
  <div class="section">
    <h2>Visual Effect Demo</h2>
    <p>Center {crop_size}×{crop_size} crop of <code>{image_name}</code>. The top row shows the original,
    traditional (JPEG/WebP/AVIF), and learned (CompressAI) reconstructions. The bottom row shows
    absolute error heatmaps for lossy codecs and the PNG lossless reference.</p>
    <div class="chart-card">
      <img src="demo_comparison.png" alt="Compression Effect Demo">
    </div>
  </div>
"""
