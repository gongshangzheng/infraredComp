"""Contour extractors. Importing this package registers all built-in methods."""

from .base import (  # noqa: F401
    ContourExtractor,
    EXTRACTOR_REGISTRY,
    build_extractor,
    list_extractors,
    register,
)

# Importing these modules triggers @register(...) on the classes.
from . import canny as _canny  # noqa: F401
from . import sobel as _sobel  # noqa: F401
from . import hed as _hed  # noqa: F401
from . import pidinet as _pidinet  # noqa: F401
from . import yoloe26 as _yoloe26  # noqa: F401

__all__ = [
    "ContourExtractor",
    "EXTRACTOR_REGISTRY",
    "build_extractor",
    "list_extractors",
    "register",
]
