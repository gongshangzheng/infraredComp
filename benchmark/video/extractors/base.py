"""Pluggable contour (edge) extractors.

A contour extractor takes a grayscale frame (uint8 HxW) and returns a grayscale
edge-intensity frame (uint8 HxW). New methods register via ``@register("name")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

# name -> extractor class
EXTRACTOR_REGISTRY: dict[str, type["ContourExtractor"]] = {}


def register(name: str) -> Callable[[type], type]:
    """Class decorator: register a ContourExtractor subclass under ``name``."""

    def _wrap(cls: type) -> type:
        if not issubclass(cls, ContourExtractor):
            raise TypeError(f"{cls.__name__} must inherit ContourExtractor")
        EXTRACTOR_REGISTRY[name] = cls
        return cls

    return _wrap


def build_extractor(name: str, **kwargs) -> "ContourExtractor":
    """Instantiate a registered extractor by name."""
    if name not in EXTRACTOR_REGISTRY:
        avail = ", ".join(sorted(EXTRACTOR_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown extractor '{name}'. Available: {avail}")
    return EXTRACTOR_REGISTRY[name](**kwargs)


def list_extractors() -> list[str]:
    """Return registered extractor names."""
    return sorted(EXTRACTOR_REGISTRY)


class ContourExtractor(ABC):
    """Abstract base for a frame-wise edge extractor."""

    name: str = "base"

    @abstractmethod
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Return a uint8 edge-intensity frame (HxW) for the given input frame.

        ``frame`` is a COLOR BGR (HxWx3) uint8 array as of the colorized stage1
        pipeline (gray sources decode to 3 identical channels — harmless). Each
        extractor decides gray-vs-color: classical ops (canny/sobel) cvtColor to
        gray; deep detectors (hed/pidinet/yoloe26) use the color directly (they
        were trained on color). Output is always single-channel uint8 HxW.
        """
        raise NotImplementedError

    def extract_video(self, src_path, fps: float | None = None,
                      frames: int | None = None) -> list[np.ndarray]:
        """Decode ``src_path`` (a video file OR a frame directory) to color frames
        and return a list of uint8 HxW edge frames (one per source frame).

        **Default = per-frame**: demux the video (or glob the dir) to color PNGs,
        read each as BGR, call ``self.extract``. The frame-splitting + per-frame
        loop lives in the MODEL so that a model which can **natively process video**
        (e.g. using temporal context) overrides this to take the whole video at
        once — it is not forced into per-frame processing. Native-video overrides
        still return per-frame edge maps (so the downstream contour-mp4 assembly is
        shared). ``fps`` is ignored by the default (per-frame needs none); overrides
        may use it for temporal reasoning.

        ``src_path`` resolves to a video (decoded via ffmpeg to bgr24 PNGs in a
        model-owned temp dir, cleaned up after) or a directory of image frames.
        """
        import shutil
        import tempfile
        from pathlib import Path

        import cv2

        from ..ffmpeg_util import demux_to_frames

        p = Path(src_path)
        _img_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        tmp = None
        if p.is_dir():
            frame_paths = sorted(f for f in p.iterdir() if f.suffix.lower() in _img_exts)
            if frames is not None:
                frame_paths = frame_paths[:frames]
        else:  # video file
            tmp = Path(tempfile.mkdtemp(prefix="extvid_"))
            try:
                frame_paths = demux_to_frames(p, tmp, frames=frames)
            except Exception:
                shutil.rmtree(tmp, ignore_errors=True)
                raise

        edges: list[np.ndarray] = []
        for fp in frame_paths:
            frame = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            e = self.extract(frame)
            if e.ndim == 3:
                e = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
            edges.append(e)

        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
        return edges
