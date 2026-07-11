"""H.265 / HEVC via libx265."""

from .base import VideoCodec, register_codec


@register_codec("x265")
class X265Codec(VideoCodec):
    name = "x265"
    family = "hevc"
    encoder = "libx265"
    default_preset = "medium"
    ext = "mp4"
