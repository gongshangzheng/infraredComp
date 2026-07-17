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


_CODEC_MODULE: dict[str, str] = {
    "x264": ".x264", "x265": ".x265", "svtav1": ".svtav1", "vp9": ".vp9", "mpeg4": ".mpeg4",
    "ssf2020": ".ssf2020", "dcvc_rt": ".dcvc_rt", "nevc": ".nevc", "dcvc_dc": ".dcvc_dc",
    "lsmc": ".lsmc", "difftok": ".difftok",
}


def build_codec(name: str, crf: int, preset: str | None = None, checkpoint_path: str | None = None) -> "VideoCodec":
    # Lazy import: if codec not yet registered, import its module to trigger @register_codec.
    # Avoids loading torch/compressai/ultralytics at catalog() time (only on actual build).
    if name not in CODEC_REGISTRY:
        import importlib
        mod = _CODEC_MODULE.get(name) or (".learned_image" if name.startswith("img-") else None)
        if mod:
            importlib.import_module(mod, __package__)
    if name not in CODEC_REGISTRY:
        avail = ", ".join(sorted(CODEC_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown codec '{name}'. Available: {avail}")
    cls = CODEC_REGISTRY[name]
    kw = {"crf": crf, "preset": preset}
    if checkpoint_path:
        kw["checkpoint_path"] = checkpoint_path  # learned codec __init__ 接；传统 codec 无此参数
    return cls(**kw)


def list_codecs() -> list[str]:
    return sorted(CODEC_REGISTRY)


@dataclass
class CodecConfig:
    crf: int
    preset: str | None = None
    pix_fmt: str = config.ENCODE_PIX_FMT
    extra_encode_args: list[str] = field(default_factory=list)


class VideoCodec:
    """Base video codec. Subclasses set encoder / default_preset / extra args.

    Two execution paths:
    - **ffmpeg** (default, ``is_neural=False``): subclass overrides ``encode_args``/``decode_args``
      to return ffmpeg argv; the harness shells out via ``run_ffmpeg``.
    - **neural in-process** (``is_neural=True``): subclass overrides ``encode_inprocess``/``decode_inprocess``
      to encode/decode a frame sequence in-process (loads a .pth, no ffmpeg). Used by learned
      video codecs (ssf2020 / dcvc-rt). The harness writes returned bytes to the bitstream file
      (for size accounting) and decoded frames to recon_dir as PNGs so the metrics pipeline is shared.
    """

    name: str = "base"
    family: str = "base"
    encoder: str = ""           # ffmpeg encoder name, e.g. libx264
    default_preset: str | None = None
    ext: str = "mp4"            # output container
    is_neural: bool = False     # True → harness uses encode_inprocess/decode_inprocess (no ffmpeg)
    # Whether the produced bitstream is directly <video>-playable in browsers.
    # False (e.g. MPEG-4 Part 2, which no browser decodes) → harness synthesizes a
    # viewable H.264 mp4 from the recon frames after metrics (size already measured).
    browser_playable: bool = True

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

    # ----- neural in-process path (learned video codecs override these) -----

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        """Encode a frame sequence in-process (no ffmpeg). frames = list[np.ndarray HxW or HxWxC uint8].

        Subclasses (ssf2020 / dcvc-rt) override: load model, run model.encode_keyframe/encode_inter,
        serialize strings+shapes to bytes. Default raises — only neural codecs (is_neural=True) implement.
        """
        raise NotImplementedError(f"{self.name} is not a neural codec (is_neural={self.is_neural})")

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int, hw: tuple[int, int]) -> list:
        """Decode bitstream bytes back to a list of np.ndarray frames (HxW uint8).

        Subclasses override: deserialize, model.decode_keyframe/decode_inter, return frames.
        """
        raise NotImplementedError(f"{self.name} is not a neural codec (is_neural={self.is_neural})")
