"""Canny edge extractor (Gaussian blur + Canny)."""

import numpy as np
import cv2

from .base import ContourExtractor, register


@register("canny")
class CannyExtractor(ContourExtractor):
    """Gaussian-blurred Canny edges.

    Parameters
    ----------
    blur_ksize : int
        Gaussian blur kernel size (must be odd). 0 disables blur.
    t1, t2 : int
        Canny hysteresis thresholds.
    """

    name = "canny"

    def __init__(self, blur_ksize: int = 5, t1: int = 50, t2: int = 150):
        self.blur_ksize = blur_ksize
        self.t1 = t1
        self.t2 = t2

    def extract(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = _to_uint8(frame)
        # stage1 now passes COLOR (bgr24) frames; canny runs on luminance.
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize > 0:
            frame = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        edges = cv2.Canny(frame, self.t1, self.t2)
        return edges  # uint8, values in {0, 255}


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 via min-max scaling."""
    if arr.dtype == np.uint8:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
