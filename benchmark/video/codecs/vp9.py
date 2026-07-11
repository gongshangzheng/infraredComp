"""VP9 via libvpx-vp9."""

from .base import VideoCodec, register_codec


@register_codec("vp9")
class Vp9Codec(VideoCodec):
    name = "vp9"
    family = "vp9"
    encoder = "libvpx-vp9"
    default_preset = None
    ext = "webm"

    def encode_args(self, frames_dir, fps, bitstream):  # type: ignore[override]
        # VP9 uses -crf too (0-63); -b:v 0 lets CRF drive rate control.
        args = super().encode_args(frames_dir, fps, bitstream)
        # Insert "-b:v 0" right before the output path (last element).
        args = args[:-1] + ["-b:v", "0", args[-1]]
        return args
