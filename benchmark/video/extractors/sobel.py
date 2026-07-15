"""Sobel edge extractor (Sobel magnitude, normalized to uint8)."""

import numpy as np
import cv2

from .base import ContourExtractor, register


@register("sobel")
class SobelExtractor(ContourExtractor):
    """Sobel gradient-magnitude edges, scaled to fill [0, 255]."""

    name = "sobel"

    def __init__(self, ksize: int = 3, blur_ksize: int = 3):
        self.ksize = ksize
        self.blur_ksize = blur_ksize

    def extract(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = _to_uint8(frame)
        # stage1 now passes COLOR (bgr24) frames; Sobel runs on luminance.
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize and self.blur_ksize > 0:
            frame = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=self.ksize)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=self.ksize)
        mag = cv2.magnitude(gx, gy)
        mn, mx = float(mag.min()), float(mag.max())
        if mx - mn > 0:
            mag = (mag - mn) / (mx - mn) * 255.0
        else:
            mag = np.zeros_like(mag)
        return mag.astype(np.uint8)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
