"""H.264 / AVC via libx264."""

from .base import VideoCodec, register_codec


@register_codec("x264")
class X264Codec(VideoCodec):
    name = "x264"
    family = "h264"
    encoder = "libx264"
    default_preset = "medium"
    ext = "mp4"
