"""Image compression benchmark — traditional codecs (JPEG, WebP, PNG, JPEG2000, AVIF, HEIC)."""

import io
from typing import Callable

import numpy as np
from PIL import Image

from .metrics import CompressionResult, compute_psnr, compute_ssim, timed


def _to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image, handling 8-bit and 16-bit."""
    if img.dtype == np.uint16:
        return Image.fromarray(img, mode="I;16")
    elif img.dtype == np.uint8:
        if img.ndim == 2:
            return Image.fromarray(img, mode="L")
        else:
            return Image.fromarray(img, mode="RGB")
    else:
        # Normalize float arrays to uint8
        arr = (img * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L" if img.ndim == 2 else "RGB")


def _to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(img)


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize a 16-bit or float image to 8-bit using min-max scaling."""
    if img.dtype == np.uint8:
        return img
    img_min, img_max = img.min(), img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)


def _compress_pil(
    original: np.ndarray,
    image_name: str,
    codec_name: str,
    pil_format: str,
    quality: int | None = None,
    lossless: bool = False,
) -> CompressionResult:
    """Generic compression via PIL save/load round-trip."""
    # Formats that only support 8-bit input
    _8BIT_FORMATS = ("JPEG", "WEBP", "AVIF", "HEIF")
    work_img = original
    if pil_format in _8BIT_FORMATS:
        work_img = _normalize_to_uint8(original)

    pil_img = _to_pil(work_img)

    # Encode
    buf = io.BytesIO()
    save_kwargs: dict = {"format": pil_format}
    if quality is not None:
        save_kwargs["quality"] = quality
    if pil_format in ("WEBP", "AVIF"):
        save_kwargs["lossless"] = lossless
        if lossless:
            save_kwargs.pop("quality", None)

    def _encode():
        buf.seek(0)
        buf.truncate()
        pil_img.save(buf, **save_kwargs)

    _, encode_ms = timed(_encode)
    compressed_bytes = buf.getvalue()

    # Decode
    def _decode():
        buf.seek(0)
        return Image.open(buf)

    decoded_pil, decode_ms = timed(_decode)
    decoded = _to_numpy(decoded_pil)

    # Align channel dimensions: WebP may return RGB for grayscale input
    if work_img.ndim == 2 and decoded.ndim == 3:
        # Convert decoded RGB back to grayscale
        decoded = decoded[:, :, 0]  # all channels identical for grayscale
    elif decoded.ndim == 2 and work_img.ndim == 3:
        decoded = np.stack([decoded, decoded, decoded], axis=-1)

    # Align dtypes for metric computation
    _8BIT_FORMATS = ("JPEG", "WEBP", "AVIF", "HEIF")
    ref = _normalize_to_uint8(original) if pil_format in _8BIT_FORMATS else original
    if ref.dtype != decoded.dtype:
        if decoded.dtype == np.uint8 and ref.dtype == np.uint16:
            decoded = decoded.astype(np.uint16) * 256
        elif ref.dtype == np.uint8 and decoded.dtype == np.uint16:
            ref = ref.astype(np.uint16) * 256

    psnr = compute_psnr(ref, decoded)
    ssim = compute_ssim(ref, decoded)
    pixel_count = original.size
    bpp = (len(compressed_bytes) * 8) / pixel_count
    original_bytes = original.nbytes
    compression_ratio = original_bytes / len(compressed_bytes) if compressed_bytes else 0

    return CompressionResult(
        codec=codec_name,
        image_name=image_name,
        psnr=psnr,
        ssim=ssim,
        bpp=bpp,
        compression_ratio=compression_ratio,
        encode_time_ms=encode_ms,
        decode_time_ms=decode_ms,
        original_shape=original.shape,
        compressed_bytes=len(compressed_bytes),
        decoded=decoded,
    )


# ---------------------------------------------------------------------------
# Public codec API
# ---------------------------------------------------------------------------

JPEG_QUALITIES = [95, 75, 50, 25]
WEBP_QUALITIES = [95, 75, 50, 25]
AVIF_QUALITIES = [95, 75, 50, 25]
HEIC_QUALITIES = [95, 75, 50, 25]


def compress_jpeg(img: np.ndarray, name: str, quality: int = 75) -> CompressionResult:
    """JPEG compression at given quality level."""
    return _compress_pil(img, name, f"JPEG q{quality}", "JPEG", quality=quality)


def compress_webp(img: np.ndarray, name: str, quality: int = 75) -> CompressionResult:
    """WebP lossy compression at given quality level."""
    return _compress_pil(img, name, f"WebP q{quality}", "WEBP", quality=quality)


def compress_webp_lossless(img: np.ndarray, name: str) -> CompressionResult:
    """WebP lossless compression."""
    return _compress_pil(img, name, "WebP-lossless", "WEBP", lossless=True)


def compress_png(img: np.ndarray, name: str) -> CompressionResult:
    """PNG lossless compression (supports 16-bit)."""
    return _compress_pil(img, name, "PNG", "PNG")


def compress_jpeg2000(img: np.ndarray, name: str) -> CompressionResult:
    """JPEG2000 lossless compression."""
    return _compress_pil(img, name, "JPEG2000-lossless", "JPEG2000")


def compress_jpeg2000_lossy(img: np.ndarray, name: str, quality: int = 75) -> CompressionResult:
    """JPEG2000 lossy compression."""
    return _compress_pil(img, name, f"JPEG2000 q{quality}", "JPEG2000", quality=quality)


def compress_avif(img: np.ndarray, name: str, quality: int = 75) -> CompressionResult:
    """AVIF lossy compression."""
    return _compress_pil(img, name, f"AVIF q{quality}", "AVIF", quality=quality)


def compress_avif_lossless(img: np.ndarray, name: str) -> CompressionResult:
    """AVIF lossless compression."""
    return _compress_pil(img, name, "AVIF-lossless", "AVIF", lossless=True)


def compress_heic(img: np.ndarray, name: str, quality: int = 75) -> CompressionResult:
    """HEIC lossy compression."""
    return _compress_pil(img, name, f"HEIC q{quality}", "HEIF", quality=quality)


def get_traditional_codecs() -> list[Callable]:
    """Return list of all traditional codec functions with their quality variants."""
    codecs = []
    for q in JPEG_QUALITIES:
        codecs.append(lambda img, name, q=q: compress_jpeg(img, name, quality=q))
    for q in WEBP_QUALITIES:
        codecs.append(lambda img, name, q=q: compress_webp(img, name, quality=q))
    codecs.append(compress_webp_lossless)
    codecs.append(compress_png)
    codecs.append(compress_jpeg2000)
    codecs.append(lambda img, name: compress_jpeg2000_lossy(img, name, quality=75))
    return codecs
