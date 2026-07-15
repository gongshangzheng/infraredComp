"""HED (Holistically-Nested Edge Detection) extractor — deep edge detector.

Runs the original `s9xie/hed` trained model (``deploy.prototxt`` +
``hed_pretrained_bsds.caffemodel``, the Berkeley BSDS release) via OpenCV's DNN
module — **no Caffe install needed**. Reuses the exact trained weights from that
project, only swapping the (abandoned) Caffe runtime for ``cv2.dnn``.

Produces a soft edge-probability map (uint8, 0..255), complementing the classical
baselines: ``canny`` (binary hard-thresholded edges) and ``sobel`` (gradient
magnitude). Same interface — ``extract(frame) -> uint8 HxW``.

Two gotchas handled here (from the OpenCV HED sample):
  1. ``deploy.prototxt`` uses a Caffe ``Crop`` layer (centers each deconv
     side-output onto the input dims). ``cv2.dnn`` has no built-in Crop layer, so
     we register a custom ``CropLayer`` before loading the net.
  2. The prototxt's final layer is ``sigmoid-fuse`` — ``net.forward()`` already
     returns edge probabilities in [0, 1]; do NOT sigmoid again, just ×255.

Weights must be pre-cached (run ``scripts/download_hed_weights.py`` for the
prototxt; the caffemodel is not on GitHub — fetch ``hed_pretrained_bsds.caffemodel``
from the Berkeley HED release and drop it in the cache dir). This module
pre-checks and raises a clear error if missing, mirroring the CompressAI guard.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .base import ContourExtractor, register

# Weights live in-repo under third_party/hed/ (the caffemodel is a 56 MB binary
# fetched from the Berkeley HED release; the prototxt is from s9xie/hed). Both
# are vendored rather than ~/.cache'd so the extractor is self-contained.
# hed.py = benchmark/video/extractors/hed.py -> parents[3] = repo root.
_HED_DIR = Path(__file__).resolve().parents[3] / "third_party" / "hed"
DEFAULT_PROTOTXT = str(_HED_DIR / "deploy.prototxt")
DEFAULT_CAFFEMODEL = str(_HED_DIR / "hed_pretrained_bsds.caffemodel")

# ImageNet per-channel means (BGR) — applied by blobFromImage. Exact values from
# the s9xie/hed / OpenCV HED recipe.
_IMGNET_MEAN_BGR = (104.00698793, 116.66876762, 122.67891434)


# --- custom Crop layer (centers a conv/deconv feature map onto input dims) ---
class CropLayer:
    """Caffe ``Crop`` layer for cv2.dnn.

    Two bottoms: the (larger) deconv side-output and the ``data`` blob. Crops the
    side-output to the data's HxW with a center offset. Ported verbatim from the
    OpenCV HED sample.
    """

    def __init__(self, params, blobs):  # noqa: D401  (cv2.dnn signature)
        self.x1 = self.y1 = self.x2 = self.y2 = 0

    def getMemoryShapes(self, inputs):
        input_shape, target_shape = inputs[0], inputs[1]
        batch, channels = input_shape[0], input_shape[1]
        height, width = target_shape[2], target_shape[3]
        self.x1 = int((input_shape[3] - width) / 2)
        self.y1 = int((input_shape[2] - height) / 2)
        self.x2 = self.x1 + width
        self.y2 = self.y1 + height
        return [[batch, channels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.y1:self.y2, self.x1:self.x2]]


_CROP_REGISTERED = False


def _ensure_crop_layer_registered() -> None:
    """Register the custom Crop layer once (idempotent across instances)."""
    global _CROP_REGISTERED
    if _CROP_REGISTERED:
        return
    try:
        cv2.dnn_registerLayer("Crop", CropLayer)
    except Exception:  # noqa: BLE001  — already registered in this process
        pass
    _CROP_REGISTERED = True


@register("hed")
class HedExtractor(ContourExtractor):
    """Holistically-Nested Edge Detection via cv2.dnn + s9xie/hed weights."""

    name = "hed"

    def __init__(
        self,
        prototxt: str = DEFAULT_PROTOTXT,
        caffemodel: str = DEFAULT_CAFFEMODEL,
    ):
        if not (os.path.isfile(prototxt) and os.path.isfile(caffemodel)):
            raise FileNotFoundError(
                "HED weights not found. Fetch deploy.prototxt with:\n"
                "    python scripts/download_hed_weights.py\n"
                "and place hed_pretrained_bsds.caffemodel alongside it in:\n"
                f"    {_HED_DIR}\n"
                f"  prototxt:   {prototxt}\n"
                f"  caffemodel: {caffemodel}"
            )
        self.prototxt = prototxt
        self.caffemodel = caffemodel
        _ensure_crop_layer_registered()
        # readNet auto-detects Caffe format from the prototxt; readNetFromCaffe
        # works too once the Crop layer is registered.
        self.net = cv2.dnn.readNet(prototxt, caffemodel)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:  # noqa: BLE001
            pass

    def extract(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            frame = _to_uint8(frame)
        # HED expects 3-channel BGR. The contour pipeline is single-channel gray;
        # expand to BGR losslessly (the network only cares about luminance edges).
        bgr = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 2
            else frame
        )
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            bgr, scalefactor=1.0, size=(w, h),
            mean=_IMGNET_MEAN_BGR, swapRB=False, crop=False,
        )
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 1, Hs, Ws) — already sigmoid'd by 'sigmoid-fuse'
        edges = out[0, 0]
        if edges.shape != (h, w):
            edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
        # forward() returns probabilities in [0, 1] → scale to uint8.
        return (np.clip(edges, 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 via min-max scaling."""
    if arr.dtype == np.uint8:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
