"""DCVC-RT (CVPR 2025 real-time neural video codec) learned video codec.

In-process neural codec (no ffmpeg): wraps microsoft/DCVC top-level models
(``DMCI`` I-frame + ``DMC`` P-frame). Encodes a contour frame sequence via
``DMCI.compress`` (I) + ``DMC.compress`` (P) loop, seeds the P-frame DPB with
the I-recon (``clear_dpb`` + ``add_ref_frame(None, x_hat)``), and serializes the
real arithmetic-coded bitstreams (rans C++ ext) to a binary container. Decodes
via ``DMCI.decompress`` / ``DMC.decompress`` mirroring the DPB seeding.

``crf`` is reused as the DCVC quantization parameter (qp). The benchmark sweeps
crf x264-style (higher crf = lower quality = fewer bits), so crf maps 1:1 to the
DCVC qp range 0..63 (qp 0 = highest quality, 63 = lowest). This keeps the RD-curve
direction consistent with x264/x265/svtav1/vp9. (ssf2020 is the opposite: its crf
is a 1..9 quality where higher = more bits — do not expect the two to align at the
same crf value.)

DCVC-RT pads to ÷16 (``DMCI.get_padding_size(h, w, 16)`` + ``replicate_pad``),
NOT ÷64. Frame I/O reuses ``benchmark.learned._img_to_tensor`` (gray→3ch,
min-max norm to [0,1]) / ``_tensor_to_img`` (denorm back to uint8); the
per-frame min/max stats are serialized so decode can reverse the normalization
without the original frames.

Setup is REQUIRED and NOT auto-installed (see ``.claude/skills/dcvc-rt-usage``):
``DCVC_REPO_ROOT`` env var must point at a cloned microsoft/DCVC, the rans C++
ext (``MLCodec_extensions_cpp``, from ``src/cpp``) must be built, and the two
CVPR-2025 checkpoints must be placed in ``<repo>/checkpoints/``. If any are
missing, ``_load`` raises a clear ``RuntimeError`` with instructions rather than
crashing at import time.

Inference-only: Microsoft has released NO training code for DCVC-RT, so contour
fine-tuning would require custom RD-loss training work (future). Override the
pretrained checkpoints via ``checkpoint_i`` / ``checkpoint_p`` (e.g. to eval a
contour-finetuned variant once one exists).
"""
from __future__ import annotations

import os
import pickle
import struct
import sys

import numpy as np
import torch

from .base import VideoCodec, register_codec
from .. import config  # noqa: F401  (keeps parity with legacy codecs re: config import)
from benchmark.learned import _img_to_tensor, _tensor_to_img


# Binary container format:
#   magic(8) + n(uint32) + h(uint32) + w(uint32)
#   per frame: type(uint8) + qp(int32) + blen(uint32) + bit_stream(blen bytes)
#   stats_len(uint32) + stats_pickle(stats_len bytes)
_MAGIC = b"DCVCRT10"
_HDR = struct.Struct("<III")      # n, h, w
_FREC = struct.Struct("<BiI")     # type, qp, blen
_LEN = struct.Struct("<I")        # stats_pickle length


@register_codec("dcvc_rt")
class DCVCRTCodec(VideoCodec):
    """DCVC-RT (microsoft/DCVC, CVPR 2025) learned video codec (in-process, neural)."""

    name = "dcvc_rt"
    family = "learned-video"
    ext = "bin"            # bitstream = serialized rans bytes (not a media container)
    is_neural = True

    _CACHE: dict = {}

    def __init__(
        self,
        crf: int,
        preset: str | None = None,
        checkpoint_i: str | None = None,
        checkpoint_p: str | None = None,
    ):
        super().__init__(crf=crf, preset=preset)
        # crf reused as DCVC qp 1:1. qp 0 = highest quality, 63 = lowest. This
        # matches the benchmark's x264-style sweep (higher crf = lower quality =
        # fewer bits) so RD-curve direction is consistent with x264/x265/svtav1/vp9.
        self.qp = max(0, min(63, int(crf)))
        self.checkpoint_i = checkpoint_i
        self.checkpoint_p = checkpoint_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = None            # lazy-load on first encode/decode

    # ---- setup error helpers ----

    @staticmethod
    def _setup_error(reason: str) -> RuntimeError:
        msg = (
            f"DCVC-RT setup incomplete ({reason}). Required setup:\n"
            "  1. Clone microsoft/DCVC (CVPR 2025 top-level, MIT license).\n"
            "  2. export DCVC_REPO_ROOT=/path/to/DCVC\n"
            "  3. Build the rans C++ ext (NO fallback): "
            "cd $DCVC_REPO_ROOT/src/cpp && pip install .  "
            "(needs cmake/g++/ninja + pybind11; installs MLCodec_extensions_cpp)\n"
            "  4. (optional, CUDA only) fused ext: "
            "cd $DCVC_REPO_ROOT/src/layers/extensions/inference && pip install .  "
            "(installs inference_extensions_cuda; auto-falls-back to pytorch if absent)\n"
            "  5. Download CVPR-2025 checkpoints from OneDrive (manual, NOT scriptable) "
            "into $DCVC_REPO_ROOT/checkpoints/:\n"
            "       cvpr2025_image.pth.tar  (DMCI, I-frame)\n"
            "       cvpr2025_video.pth.tar  (DMC,  P-frame)\n"
            "See .claude/skills/dcvc-rt-usage for full instructions.\n"
        )
        return RuntimeError(msg)

    def _load(self) -> dict:
        key = (self.device, self.checkpoint_i, self.checkpoint_p)
        if key in DCVCRTCodec._CACHE:
            return DCVCRTCodec._CACHE[key]

        repo = os.environ.get("DCVC_REPO_ROOT")
        if not repo or not os.path.isdir(repo):
            raise self._setup_error("DCVC_REPO_ROOT env var not set or path missing")

        # rans C++ ext — required (real arithmetic-coded bitstreams), no fallback.
        try:
            import MLCodec_extensions_cpp  # noqa: F401
        except ImportError as exc:
            raise self._setup_error(
                f"rans ext MLCodec_extensions_cpp not importable ({exc})"
            ) from exc

        if repo not in sys.path:
            sys.path.insert(0, repo)

        from src.models.image_model import DMCI
        from src.models.video_model import DMC
        from src.utils.common import get_state_dict
        from src.layers.cuda_inference import replicate_pad

        ckpt_i = self.checkpoint_i or os.path.join(
            repo, "checkpoints", "cvpr2025_image.pth.tar"
        )
        ckpt_p = self.checkpoint_p or os.path.join(
            repo, "checkpoints", "cvpr2025_video.pth.tar"
        )
        for label, p in (("I-frame", ckpt_i), ("P-frame", ckpt_p)):
            if not os.path.isfile(p):
                raise self._setup_error(f"{label} checkpoint not found: {p}")

        # I-frame net (DMCI)
        i_net = DMCI()
        i_net.load_state_dict(get_state_dict(ckpt_i))
        i_net = i_net.to(self.device).eval()
        i_net.update(force_zero_thres=0.12)      # README-recommended value

        # P-frame net (DMC)
        p_net = DMC()
        p_net.load_state_dict(get_state_dict(ckpt_p))
        p_net = p_net.to(self.device).eval()
        p_net.update(force_zero_thres=0.12)

        # fp16 on CUDA (faster); stay fp32 on CPU (half not supported/beneficial).
        if self.device == "cuda":
            i_net = i_net.half()
            p_net = p_net.half()

        bundle = {
            "i_net": i_net,
            "p_net": p_net,
            "DMCI": DMCI,
            "replicate_pad": replicate_pad,
        }
        DCVCRTCodec._CACHE[key] = bundle
        return bundle

    @property
    def models(self) -> dict:
        if self._models is None:
            self._models = self._load()
        return self._models

    # ---- frame I/O helpers ----

    def _make_sps(self, h: int, w: int, use_two_ec: bool) -> dict:
        return {
            "sps_id": 0,
            "height": h,
            "width": w,
            "ec_part": 1 if use_two_ec else 0,
            "use_ada_i": 0,                     # no feature-adaptor-i reset (constant qp)
        }

    # ---- in-process encode/decode ----

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        """frames = list[np.ndarray HxW uint8] -> bytes (binary container)."""
        bundle = self.models
        i_net = bundle["i_net"]
        p_net = bundle["p_net"]
        DMCI = bundle["DMCI"]
        replicate_pad = bundle["replicate_pad"]

        h0, w0 = frames[0].shape[:2]
        pad_r, pad_b = DMCI.get_padding_size(h0, w0, 16)
        use_two_ec = (h0 * w0) > (1280 * 720)
        i_net.set_use_two_entropy_coders(use_two_ec)
        p_net.set_use_two_entropy_coders(use_two_ec)
        p_net.set_curr_poc(0)

        qp = self.qp
        frame_records: list[tuple[int, int, bytes]] = []   # (type, qp, bit_stream)
        stats_list: list[dict] = []

        with torch.no_grad():
            # frame 0: I-frame (DMCI). Returns bit_stream + x_hat (recon).
            x0, st0 = _img_to_tensor(frames[0])
            x0 = x0.to(self.device)
            if self.device == "cuda":
                x0 = x0.half()
            x0_pad = replicate_pad(x0, pad_b, pad_r)
            enc = i_net.compress(x0_pad, qp)
            # Seed the P-frame DPB with the I-recon (frame, not feature).
            p_net.clear_dpb()
            p_net.add_ref_frame(None, enc["x_hat"])
            frame_records.append((0, qp, bytes(enc["bit_stream"])))
            stats_list.append(st0)

            # frames 1..n-1: P-frames (DMC). compress() internally add_ref_frame,
            # so the P-DPB advances itself; we only seeded the I-recon above.
            for i in range(1, len(frames)):
                xi, sti = _img_to_tensor(frames[i])
                xi = xi.to(self.device)
                if self.device == "cuda":
                    xi = xi.half()
                xi_pad = replicate_pad(xi, pad_b, pad_r)
                enc = p_net.compress(xi_pad, qp)
                frame_records.append((1, qp, bytes(enc["bit_stream"])))
                stats_list.append(sti)

        return self._serialize(frame_records, len(frames), h0, w0, stats_list)

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int, hw: tuple[int, int]) -> list:
        """bytes -> list[np.ndarray HxW uint8] (decoded contour frames)."""
        bundle = self.models
        i_net = bundle["i_net"]
        p_net = bundle["p_net"]

        records, (h0, w0), stats_list = self._deserialize(bitstream_bytes)
        if len(records) != n_frames:
            raise RuntimeError(
                f"DCVC-RT bitstream frame count {len(records)} != requested {n_frames}"
            )

        use_two_ec = (h0 * w0) > (1280 * 720)
        i_net.set_use_two_entropy_coders(use_two_ec)
        p_net.set_use_two_entropy_coders(use_two_ec)
        p_net.set_curr_poc(0)
        sps = self._make_sps(h0, w0, use_two_ec)

        recons: list[np.ndarray] = []
        with torch.no_grad():
            for idx, (ftype, qp, bs) in enumerate(records):
                if ftype == 0:  # I-frame
                    dec = i_net.decompress(bs, sps, qp)
                    x_hat = dec["x_hat"]
                    p_net.clear_dpb()
                    p_net.add_ref_frame(None, x_hat)
                else:           # P-frame
                    dec = p_net.decompress(bs, sps, qp)
                    x_hat = dec["x_hat"]
                # crop padding, fp16->fp32 for safe numpy/uint8 conversion, denorm.
                x_hat = x_hat[:, :, :h0, :w0].float()
                recons.append(_tensor_to_img(x_hat, stats_list[idx]))
        return recons

    # ---- (de)serialization ----

    @staticmethod
    def _serialize(
        records: list[tuple[int, int, bytes]],
        n: int,
        h: int,
        w: int,
        stats_list: list[dict],
    ) -> bytes:
        out = bytearray()
        out += _MAGIC
        out += _HDR.pack(n, h, w)
        for ftype, qp, bs in records:
            out += _FREC.pack(ftype, qp, len(bs))
            out += bs
        stats_pickle = pickle.dumps(stats_list)
        out += _LEN.pack(len(stats_pickle))
        out += stats_pickle
        return bytes(out)

    @staticmethod
    def _deserialize(buf: bytes) -> tuple[list[tuple[int, int, bytes]], tuple[int, int], list[dict]]:
        if not buf.startswith(_MAGIC):
            raise RuntimeError(
                f"DCVC-RT bitstream magic mismatch (got {buf[:8]!r}); not a dcvc_rt stream"
            )
        off = len(_MAGIC)
        n, h, w = _HDR.unpack_from(buf, off)
        off += _HDR.size
        records: list[tuple[int, int, bytes]] = []
        for _ in range(n):
            ftype, qp, blen = _FREC.unpack_from(buf, off)
            off += _FREC.size
            bs = buf[off:off + blen]
            off += blen
            records.append((ftype, qp, bs))
        (stats_len,) = _LEN.unpack_from(buf, off)
        off += _LEN.size
        stats_list = pickle.loads(buf[off:off + stats_len])
        return records, (h, w), stats_list
