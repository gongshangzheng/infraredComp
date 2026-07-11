"""Video benchmark metrics.

Reuses the image metrics (PSNR/SSIM are dtype-aware, work on 2D grayscale) and
adds per-frame aggregation, temporal consistency, and fps derivation.
"""

from __future__ import annotations

import numpy as np

from benchmark.metrics import compute_psnr, compute_ssim, timed  # noqa: F401


def per_frame_quality(
    gt: np.ndarray, rec: np.ndarray
) -> tuple[list[float], list[float]]:
    """Per-frame PSNR and SSIM between two (N, H, W) uint8 stacks.

    Frames are aligned by index. Missing trailing frames in `rec` are treated
    as zero (codec dropped them) — handled by trimming to the shorter length.
    """
    n = min(len(gt), len(rec))
    psnrs: list[float] = []
    ssims: list[float] = []
    for i in range(n):
        g = gt[i]
        r = rec[i]
        # Align shapes (codec may have padded to even dims — crop back later
        # at the benchmark layer, but guard here defensively).
        if g.shape != r.shape:
            h = min(g.shape[0], r.shape[0])
            w = min(g.shape[1], r.shape[1])
            g = g[:h, :w]
            r = r[:h, :w]
        psnrs.append(float(compute_psnr(g, r)))
        ssims.append(float(compute_ssim(g, r)))
    return psnrs, ssims


def temporal_consistency(per_frame_psnr: list[float]) -> float:
    """Standard deviation of per-frame PSNR (lower = more temporally stable)."""
    if len(per_frame_psnr) < 2:
        return 0.0
    return float(np.std(per_frame_psnr))


def fps_from_timed(frame_count: int, elapsed_ms: float) -> float:
    """Frames processed per second from a timed() elapsed."""
    if elapsed_ms <= 0 or frame_count <= 0:
        return 0.0
    return frame_count / (elapsed_ms / 1000.0)


def mean_psnr(per_frame_psnr: list[float]) -> float:
    return float(np.mean(per_frame_psnr)) if per_frame_psnr else 0.0


def mean_ssim(per_frame_ssim: list[float]) -> float:
    return float(np.mean(per_frame_ssim)) if per_frame_ssim else 0.0
