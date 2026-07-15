"""YOLOE-26 (prompt-free) extractor — object-segmentation-boundary contour.

Wraps Ultralytics YOLOE-26 **prompt-free** seg model (``yoloe-26s-seg-pf.pt``,
SASS — Segment Any object without request). Unlike canny/sobel/hed/pidinet (dense
texture edges), this runs open-vocabulary instance segmentation and draws each
instance **mask boundary** as the contour — a semantically different, object
silhouette contour (denser than text-prompted YOLOE: ~50-60 masks/frame vs ~6).

Prompt-free needs no class list / text prompts — the ``-pf`` weight has the fused
vocabulary baked, ``predict`` works directly. **Requires COLOR input** (trained on
color photos; grayscale yields ~0 detections) — stage1 now decodes+passes color
BGR frames and lets each extractor decide gray-vs-color (see
``integrate-third-party-model`` skill).

Code via ``pip install ultralytics`` (used as-is, not vendored — we don't modify
its internals); only the ~32 MB weight is vendored under ``third_party/yoloe26/``
(gitignored). Same ``extract(frame) -> uint8 HxW`` contract.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .base import ContourExtractor, register

_YOLOE_DIR = Path(__file__).resolve().parents[3] / "third_party" / "yoloe26"
DEFAULT_WEIGHTS = str(_YOLOE_DIR / "yoloe-26s-seg-pf.pt")

# device -> YOLOE model; cached so build_extractor doesn't reload the 32 MB pth.
_CACHE: dict[str, object] = {}


def _load(device: str):
    if device in _CACHE:
        return _CACHE[device]
    from ultralytics import YOLOE
    net = YOLOE(DEFAULT_WEIGHTS)  # prompt-free seg; predict() needs no prompts
    if device != "cpu":
        try:
            net.to(device)
        except Exception:  # noqa: BLE001
            pass
    _CACHE[device] = net
    return net


@register("yoloe26")
class Yoloe26Extractor(ContourExtractor):
    """YOLOE-26 prompt-free instance-mask boundaries (uint8 0/255)."""

    name = "yoloe26"

    def __init__(self, weights: str = DEFAULT_WEIGHTS, conf: float = 0.05,
                 imgsz: int = 640):
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                "YOLOE-26 weights not found. Fetch with:\n"
                "    curl -L -o third_party/yoloe26/yoloe-26s-seg-pf.pt "
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg-pf.pt\n"
                "  (also needs `pip install ultralytics`; ~32 MB, prompt-free seg)\n"
                f"  expected: {weights}"
            )
        import torch
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = _load(self.device)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        import torch
        if frame.dtype != np.uint8:
            frame = _to_uint8(frame)
        h, w = frame.shape[:2]
        # ultralytics predict accepts a BGR ndarray (cv2 convention) directly.
        res = self.net.predict(
            frame, verbose=False, conf=self.conf, imgsz=self.imgsz,
            device=self.device,
        )[0]
        edge = np.zeros((h, w), dtype=np.uint8)
        if res.masks is None:
            return edge
        masks = res.masks.data  # (N, Hp, Wp) at prediction (letterboxed) size
        for i in range(masks.shape[0]):
            mk = masks[i].cpu().numpy().astype(np.uint8)
            if mk.shape != (h, w):
                mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(edge, cnts, -1, 255, 1)
        return edge


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
