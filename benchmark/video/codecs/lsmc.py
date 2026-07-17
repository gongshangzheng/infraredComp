"""LSMC (Lossless Segmentation Map Compression) — InterDigital C++ CLI codec.

InterDigitalInc/LosslessSegmentationMapCompression is a **pure C++17 CLI codec**
(encoder/decoder binaries) doing context-adaptive chain coding (MSC/3OT) +
arithmetic coding on raw YUV segmentation maps. **No torch, no weights, no rate
points — lossless → single RD point, PSNR=inf.** Built from source (cmake+MSVC,
CMakeLists needed 2 fixes: arithmetic_codec path + missing acodec.cpp) →
``models/lsmc/{encoder,decoder}.exe`` (gitignored binaries).

Integration: ``is_neural=True`` (in-process bytes-in/bytes-out contract) but
internally subprocesses the C++ binaries (not torch). Encode = frames → temp
``.yuv`` (gray, ``-t 400``, ``H*W*F`` bytes) → ``encoder`` → ``.bin`` bytes.
Decode = ``.bin`` → ``decoder`` → ``rec.yuv`` → parse frames. 10 MB internal
buffer limit in the C++ codec (large sequences beware).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .base import VideoCodec, register_codec

_LSMC_DIR = Path(__file__).resolve().parents[3] / "models" / "lsmc"
DEFAULT_ENCODER = str(_LSMC_DIR / "encoder.exe")
DEFAULT_DECODER = str(_LSMC_DIR / "decoder.exe")


@register_codec("lsmc")
class LsmcCodec(VideoCodec):
    """LSMC lossless segmentation-map codec (C++ subprocess, single lossless point)."""

    name = "lsmc"
    family = "lsmc"
    ext = "bin"
    is_neural = True
    browser_playable = False

    def __init__(self, crf: int = 0, preset: str | None = None,
                 encoder: str | None = None, decoder: str | None = None):
        super().__init__(crf=crf, preset=preset)
        self.encoder_bin = encoder or DEFAULT_ENCODER
        self.decoder_bin = decoder or DEFAULT_DECODER
        if not (os.path.isfile(self.encoder_bin) and os.path.isfile(self.decoder_bin)):
            raise FileNotFoundError(
                "LSMC binaries not found. Build from InterDigitalInc/"
                "LosslessSegmentationMapCompression (cmake+MSVC; CMakeLists needs "
                "arithmetic_codec path fix + acodec.cpp added) → place encoder.exe/"
                f"decoder.exe in {_LSMC_DIR}\n"
                f"  encoder: {self.encoder_bin}\n  decoder: {self.decoder_bin}"
            )

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        """frames = list[np.ndarray HxW uint8] -> .bin bytes (lossless)."""
        h, w = frames[0].shape[:2]
        n = len(frames)
        tmp_dir = tempfile.mkdtemp(prefix="lsmc_enc_")
        try:
            yuv_path = os.path.join(tmp_dir, "in.yuv")
            bin_path = os.path.join(tmp_dir, "out.bin")
            # Write raw gray YUV: H*W bytes per frame (type 400 = grayscale).
            with open(yuv_path, "wb") as f:
                for fr in frames:
                    f.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
            # encoder -i in.yuv -o out.bin -r H -c W -f F -s 0 -t 400
            r = subprocess.run(
                [self.encoder_bin, "-i", yuv_path, "-o", bin_path,
                 "-r", str(h), "-c", str(w), "-f", str(n), "-s", "0", "-t", "400"],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or not os.path.isfile(bin_path):
                raise RuntimeError(
                    f"LSMC encoder failed (code {r.returncode}):\n{r.stderr[-500:]}"
                )
            with open(bin_path, "rb") as f:
                return f.read()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int,
                        hw: tuple[int, int]) -> list:
        """bytes -> list[np.ndarray HxW uint8 (lossless reconstruction)."""
        h, w = hw
        tmp_dir = tempfile.mkdtemp(prefix="lsmc_dec_")
        try:
            bin_path = os.path.join(tmp_dir, "in.bin")
            yuv_path = os.path.join(tmp_dir, "rec.yuv")
            with open(bin_path, "wb") as f:
                f.write(bitstream_bytes)
            # decoder -i in.bin -o rec.yuv
            r = subprocess.run(
                [self.decoder_bin, "-i", bin_path, "-o", yuv_path],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or not os.path.isfile(yuv_path):
                raise RuntimeError(
                    f"LSMC decoder failed (code {r.returncode}):\n{r.stderr[-500:]}"
                )
            # Parse rec.yuv: H*W bytes per frame (type 400 grayscale).
            raw = np.fromfile(yuv_path, dtype=np.uint8)
            frame_size = h * w
            frames = []
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                if end > len(raw):
                    break
                frames.append(raw[start:end].reshape(h, w))
            return frames
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
