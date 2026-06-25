"""Image compression benchmark — metrics."""

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class CompressionResult:
    """Result of compressing a single image with a single codec."""

    codec: str
    image_name: str
    psnr: float
    ssim: float
    bpp: float  # bits per pixel (compressed size / pixel count)
    compression_ratio: float  # original bytes / compressed bytes
    encode_time_ms: float
    decode_time_ms: float
    original_shape: tuple = field(default_factory=tuple)
    compressed_bytes: int = 0
    decoded: np.ndarray | None = None  # reconstructed image for demo visualizations


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute PSNR between original and reconstructed images."""
    return peak_signal_noise_ratio(original, reconstructed)


def compute_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute SSIM between original and reconstructed images."""
    # Determine data range
    data_range = original.max() - original.min()
    if data_range == 0:
        data_range = 1.0
    # Handle multi-channel vs single-channel
    if original.ndim == 3:
        channel_axis = -1
    else:
        channel_axis = None
    return structural_similarity(
        original, reconstructed, data_range=data_range, channel_axis=channel_axis
    )


def timed(fn: Callable) -> tuple:
    """Run fn and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms
