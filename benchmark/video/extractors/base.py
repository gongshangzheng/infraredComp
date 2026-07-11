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
    def extract(self, frame_gray: np.ndarray) -> np.ndarray:
        """Return a uint8 edge-intensity frame for the given grayscale frame."""
        raise NotImplementedError
