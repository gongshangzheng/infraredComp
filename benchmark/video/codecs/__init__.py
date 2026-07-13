"""Video codecs. Importing this package registers all built-in codecs."""

from .base import (  # noqa: F401
    CodecConfig,
    VideoCodec,
    CODEC_REGISTRY,
    build_codec,
    list_codecs,
    register_codec,
)

from . import x264 as _x264  # noqa: F401
from . import x265 as _x265  # noqa: F401
from . import svtav1 as _svtav1  # noqa: F401
from . import vp9 as _vp9  # noqa: F401
from . import ssf2020 as _ssf2020  # noqa: F401

__all__ = [
    "CodecConfig",
    "VideoCodec",
    "CODEC_REGISTRY",
    "build_codec",
    "list_codecs",
    "register_codec",
]
