"""NEVC (bytedance/NEVC-1.0 = EHVC) learned video codec — DCVC-derived.

ByteDance NEVC-1.0 is **EHVC** (Efficient Hierarchical Reference and Quality
Structure for Neural Video Coding, ACM MM 2025, arXiv:2509.04118) — a fork of
microsoft/DCVC (test.py header: "modified from DCVC"). Same I/P interface as
DCVC-DC: ``IntraNoAR`` (I-frame) + ``DMC`` (P-frame) + ``MLCodec_rans`` C++ entropy
ext. Vendored under ``third_party/nevc/``; checkpoint on HuggingFace
``ByteDance/NEVC1.0`` (``nevc1.0_intra.pth.tar`` / ``nevc1.0_inter.pth.tar``) —
reachable here, auto-downloaded. The ``MLCodec_rans`` ext is **reused** from the
DCVC-DC build (cp312 ``.pyd`` copied into ``third_party/nevc/src/models/`` — the
``src/cpp`` is byte-identical to DCVC-DC's).

Differences from ``dcvc_rt`` (DCVC-RT): (1) ``IntraNoAR`` I-frame (not ``DMCI``);
(2) DMC uses an **explicit DPB dict** (seeded from the I-recon) rather than
``clear_dpb``/``add_ref_frame``; (3) ``DMC.compress`` needs ``x_next`` (look-ahead
— the next frame, ``None`` for the last); (4) rate control = ``q_in_ckpt=True`` +
``q_index`` (NEVC's rate points), not DCVC-RT's ``qp``.

Setup: ``third_party/nevc/src/`` vendored (tracked); checkpoints at
``third_party/nevc/models/nevc1.0_{intra,inter}.pth.tar`` (gitignored — fetch via
``scripts/download_nevc_weights.py`` or ``huggingface_hub``); rans ext reused
(``MLCodec_rans``/``MLCodec_CXX`` ``.pyd`` in ``src/models/``). ``_load`` raises a
clear error if any are missing. Inference-only (no training code shipped).
"""
from __future__ import annotations

import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import torch

from .base import VideoCodec, register_codec
from .. import config  # noqa: F401
from benchmark.learned import _img_to_tensor, _tensor_to_img

# Binary container: magic + n,h,w + per-frame (type, q_index, blen) + bitstream
# + stats_pickle(len+bytes). Same shape as dcvc_rt's container.
_MAGIC = b"NEVC1\x00\x00\x00"
_HDR = struct.Struct("<III")      # n, h, w
_FREC = struct.Struct("<BiI")     # type, q_index, blen
_LEN = struct.Struct("<I")        # stats_pickle length

_NEVC_DIR = Path(__file__).resolve().parents[3] / "third_party" / "nevc"
DEFAULT_CKPT_I = str(_NEVC_DIR / "models" / "nevc1.0_intra.pth.tar")
DEFAULT_CKPT_P = str(_NEVC_DIR / "models" / "nevc1.0_inter.pth.tar")

# Empty DPB after the I-frame seeds it with the I-recon (NEVC test.py shape).
def _seed_dpb(x_hat) -> dict:
    return {
        "ref_frame": x_hat,
        "ref_feature": None,
        "ref_mv_feature": None,
        "ref_y": None,
        "ref_mv_y": None,
        "key_feature": None,
        "key_mvs": [],
    }


@register_codec("nevc")
class NevcCodec(VideoCodec):
    """NEVC-1.0 (EHVC) — IntraNoAR I-frame + DMC P-frame, DCVC-derived."""

    name = "nevc"
    family = "nevc"
    ext = "bin"
    is_neural = True
    browser_playable = False
    _CACHE: dict = {}

    def __init__(self, crf: int = 1, preset: str | None = None,
                 checkpoint_i: str | None = None, checkpoint_p: str | None = None):
        super().__init__(crf=crf, preset=preset)
        self.q_index = int(crf)              # NEVC rate point (q_index)
        self.checkpoint_i = checkpoint_i
        self.checkpoint_p = checkpoint_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = None

    def _setup_error(self, reason: str) -> RuntimeError:
        return RuntimeError(
            f"NEVC setup incomplete: {reason}\n"
            "  1. vendor src: git clone --depth 1 bytedance/NEVC, copy NEVC-1.0-EHVC/src "
            "-> third_party/nevc/src\n"
            "  2. checkpoints: python scripts/download_nevc_weights.py  "
            "(HuggingFace ByteDance/NEVC1.0 -> third_party/nevc/models/)\n"
            "  3. rans ext: reuse DCVC-DC's MLCodec_rans/MLCodec_CXX cp312 .pyd into "
            "third_party/nevc/src/models/ (src/cpp is identical)\n"
            f"     I ckpt: {self.checkpoint_i or DEFAULT_CKPT_I}\n"
            f"     P ckpt: {self.checkpoint_p or DEFAULT_CKPT_P}"
        )

    def _load(self) -> dict:
        key = (self.device, self.checkpoint_i, self.checkpoint_p)
        if key in NevcCodec._CACHE:
            return NevcCodec._CACHE[key]
        repo = str(_NEVC_DIR)
        if not os.path.isdir(os.path.join(repo, "src", "models")):
            raise self._setup_error("third_party/nevc/src not found")
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # ext present? (entropy_models.py imports .MLCodec_rans / .MLCodec_CXX)
        try:
            from src.models.MLCodec_rans import RansEncoder  # noqa: F401
            from src.models.MLCodec_CXX import pmf_to_quantized_cdf  # noqa: F401
        except ImportError as exc:
            raise self._setup_error(
                f"MLCodec_rans/MLCodec_CXX ext not importable ({exc}); "
                "copy the cp312 .pyd from DCVC-DC's src/models/"
            ) from exc

        from src.models.image_model import IntraNoAR
        from src.models.video_model import DMC
        from src.utils.stream_helper import get_state_dict, get_padding_size  # NEVC: stream_helper, not common

        ckpt_i = self.checkpoint_i or DEFAULT_CKPT_I
        ckpt_p = self.checkpoint_p or DEFAULT_CKPT_P
        for label, p in (("I-frame", ckpt_i), ("P-frame", ckpt_p)):
            if not os.path.isfile(p):
                raise self._setup_error(f"{label} checkpoint not found: {p}")

        i_net = IntraNoAR(ec_thread=False, stream_part=1, inplace=True)
        i_net.load_state_dict(get_state_dict(ckpt_i))
        i_net = i_net.to(self.device).eval()
        try:
            i_net.update(force=True)
        except Exception:  # noqa: BLE001
            pass

        p_net = DMC(ec_thread=False, stream_part=1, inplace=True)
        p_net.load_state_dict(get_state_dict(ckpt_p))
        p_net = p_net.to(self.device).eval()
        try:
            p_net.update(force=True)
        except Exception:  # noqa: BLE001
            pass

        # NEVC's DMC flow_warp (grid_sample) does NOT support fp16 (Half/Float
        # mismatch in the flow grid) — stay fp32 on CUDA (slower than dcvc_rt's
        # fp16, but correct).
        bundle = {"i_net": i_net, "p_net": p_net, "get_padding_size": get_padding_size}
        NevcCodec._CACHE[key] = bundle
        return bundle

    @property
    def models(self) -> dict:
        if self._models is None:
            self._models = self._load()
        return self._models

    # ---- in-process encode/decode ----

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        """frames = list[np.ndarray HxW uint8] -> bytes (binary container)."""
        bundle = self.models
        i_net, p_net, get_padding_size = bundle["i_net"], bundle["p_net"], bundle["get_padding_size"]
        h0, w0 = frames[0].shape[:2]
        pad_l, pad_r, pad_t, pad_b = get_padding_size(h0, w0, 16)

        records: list[tuple[int, int, bytes]] = []   # (type, q_index, bit_stream)
        stats_list: list[dict] = []
        with torch.no_grad():
            # I-frame (frame 0).
            x0, st0 = _img_to_tensor(frames[0])
            x0 = x0.to(self.device)
            x0_pad = torch.nn.functional.pad(x0, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
            enc = i_net.compress(x0_pad, True, self.q_index)
            dpb = _seed_dpb(enc["x_hat"])
            records.append((0, self.q_index, bytes(enc["bit_stream"])))
            stats_list.append(st0)

            # P-frames 1..n-1 (look-ahead x_next; None for the last frame).
            for i in range(1, len(frames)):
                xi, sti = _img_to_tensor(frames[i])
                xi = xi.to(self.device)
                xi_pad = torch.nn.functional.pad(xi, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
                if i + 1 < len(frames):
                    xn, _ = _img_to_tensor(frames[i + 1])
                    xn = xn.to(self.device)
                    x_next_pad = torch.nn.functional.pad(xn, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
                else:
                    x_next_pad = None
                enc = p_net.compress(xi_pad, x_next_pad, dpb, True, self.q_index, i % 4)
                dpb = enc["dpb"]
                records.append((1, self.q_index, bytes(enc["bit_stream"])))
                stats_list.append(sti)

        return self._serialize(records, len(frames), h0, w0, stats_list)

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int, hw: tuple[int, int]) -> list:
        """bytes -> list[np.ndarray HxW uint8]."""
        bundle = self.models
        i_net, p_net, get_padding_size = bundle["i_net"], bundle["p_net"], bundle["get_padding_size"]
        records, (h0, w0), stats_list = self._deserialize(bitstream_bytes)
        if len(records) != n_frames:
            raise RuntimeError(
                f"NEVC bitstream frame count {len(records)} != requested {n_frames}"
            )
        pad_l, pad_r, pad_t, pad_b = get_padding_size(h0, w0, 16)

        recons: list[np.ndarray] = []
        with torch.no_grad():
            for idx, (ftype, q, bs) in enumerate(records):
                if ftype == 0:  # I-frame
                    dec = i_net.decompress(bs, h0, w0, True, q)
                    x_hat = dec["x_hat"]
                    dpb = _seed_dpb(x_hat)
                else:           # P-frame
                    dec = p_net.decompress(dpb, bs, h0, w0, True, q, idx % 4)
                    x_hat = dec["x_hat"]
                    dpb = dec.get("dpb", dpb)
                x_hat = x_hat[:, :, :h0, :w0].float()
                recons.append(_tensor_to_img(x_hat, stats_list[idx]))
        return recons

    # ---- (de)serialization ----

    @staticmethod
    def _serialize(records, n, h, w, stats_list) -> bytes:
        out = bytearray()
        out += _MAGIC
        out += _HDR.pack(n, h, w)
        for ftype, q, bs in records:
            out += _FREC.pack(ftype, q, len(bs))
            out += bs
        stats_pickle = pickle.dumps(stats_list)
        out += _LEN.pack(len(stats_pickle))
        out += stats_pickle
        return bytes(out)

    @staticmethod
    def _deserialize(buf):
        if not buf.startswith(_MAGIC):
            raise RuntimeError(f"NEVC bitstream magic mismatch (got {buf[:8]!r})")
        off = len(_MAGIC)
        n, h, w = _HDR.unpack_from(buf, off); off += _HDR.size
        records = []
        for _ in range(n):
            ftype, q, blen = _FREC.unpack_from(buf, off); off += _FREC.size
            records.append((ftype, q, buf[off:off + blen])); off += blen
        (slen,) = _LEN.unpack_from(buf, off); off += _LEN.size
        stats_list = pickle.loads(buf[off:off + slen])
        return records, (h, w), stats_list
