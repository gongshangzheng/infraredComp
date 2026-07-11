"""AV1 via libsvtav1."""

from .base import VideoCodec, register_codec


@register_codec("svtav1")
class SvtAv1Codec(VideoCodec):
    name = "svtav1"
    family = "av1"
    encoder = "libsvtav1"
    default_preset = None  # svtav1 uses -preset separately; CRF is primary
    ext = "mp4"

    def encode_args(self, frames_dir, fps, bitstream):  # type: ignore[override]
        # libsvtav1 prefers an explicit tier/CRF; reuse the base builder, but
        # avoid -preset defaulting to a slow path by leaving it unset.
        args = super().encode_args(frames_dir, fps, bitstream)
        return args
