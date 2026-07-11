"""Video codec abstraction.

Every codec encodes the lossless contour PNG sequence to a bitstream with a
fixed pixel format (yuv420p) and decodes back to single-channel gray PNGs, so
PSNR is comparable across codecs. Odd dimensions are padded to even at encode
time (yuv420p requires it) and cropped back before metric computation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

from .. import config

CODEC_REGISTRY: dict[str, type["VideoCodec"]] = {}


def register_codec(name: str) -> Callable[[type], type]:
    def _wrap(cls: type) -> type:
        if not issubclass(cls, VideoCodec):
            raise TypeError(f"{cls.__name__} must inherit VideoCodec")
        CODEC_REGISTRY[name] = cls
        return cls

    return _wrap


def build_codec(name: str, crf: int, preset: str | None = None) -> "VideoCodec":
    if name not in CODEC_REGISTRY:
        avail = ", ".join(sorted(CODEC_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown codec '{name}'. Available: {avail}")
    cls = CODEC_REGISTRY[name]
    return cls(crf=crf, preset=preset)


def list_codecs() -> list[str]:
    return sorted(CODEC_REGISTRY)


@dataclass
class CodecConfig:
    crf: int
    preset: str | None = None
    pix_fmt: str = config.ENCODE_PIX_FMT
    extra_encode_args: list[str] = field(default_factory=list)


class VideoCodec:
    """Base video codec. Subclasses set encoder / default_preset / extra args."""

    name: str = "base"
    family: str = "base"
    encoder: str = ""           # ffmpeg encoder name, e.g. libx264
    default_preset: str | None = None
    ext: str = "mp4"            # output container

    def __init__(self, crf: int, preset: str | None = None):
        self.cfg = CodecConfig(
            crf=crf,
            preset=preset or self.default_preset,
        )

    # ----- encode / decode argument builders -----

    def encode_args(
        self,
        frames_dir: str,
        fps: float,
        bitstream: str,
    ) -> list[str]:
        """ffmpeg args: read PNG sequence, encode to bitstream at this CRF."""
        args = [
            "-y",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%06d.png"),
        ]
        # Pad odd dims to even (yuv420p requirement). No-op for even dims.
        args += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black"]
        args += ["-c:v", self.encoder, "-crf", str(self.cfg.crf)]
        if self.cfg.preset:
            args += ["-preset", self.cfg.preset]
        args += ["-pix_fmt", self.cfg.pix_fmt]
        args += self.cfg.extra_encode_args
        args += [bitstream]
        return args

    def decode_args(self, bitstream: str, out_dir: str) -> list[str]:
        """ffmpeg args: decode bitstream to single-channel gray PNGs."""
        import os as _os
        return [
            "-y",
            "-i", bitstream,
            "-pix_fmt", config.DECODE_PIX_FMT,
            _os.path.join(out_dir, "frame_%06d.png"),
        ]
