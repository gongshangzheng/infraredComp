"""DCVC-DC (CVPR 2023) learned video codec — Microsoft DCVC family member.

DCVC-DC is the DCVC-family's CVPR 2023 variant, using the same I/P interface
as NEVC (bytedance/NEVC-1.0 = EHVC): ``IntraNoAR`` (I-frame) + ``DMC`` (P-frame)
+ ``MLCodec_rans`` C++ entropy ext. The ext is built from DCVC-DC's own ``src/cpp``
(cmake, cp312) — byte-identical to NEVC's, so the same ``.pyd`` is reused.

Checkpoints: ``cvpr2023_image_psnr.pth.tar`` + ``cvpr2023_video_psnr.pth.tar``
on OneDrive (blocked here — download manually per t10-1 description). Vendor
src under ``third_party/DCVC/DCVC-family/DCVC-DC/`` (in the submodule).

This wrapper is a near-copy of ``nevc.py`` with src + checkpoint paths adapted.
NEVC proved the ``encode_decode`` + ``decode_i``/``decode_p`` path works
(bpp=0.31, psnr=33.08). DCVC-DC has the same API.
"""
from __future__ import annotations

import os
import pickle
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from .base import VideoCodec, register_codec
from .. import config  # noqa: F401
from benchmark.learned import _img_to_tensor, _tensor_to_img

_MAGIC = b"DCVCDC10"
_HDR = struct.Struct("<III")      # n, h, w
_FREC = struct.Struct("<BiI")     # type, q_index, blen
_LEN = struct.Struct("<I")        # stats_pickle length

_DCVC_DC_DIR = Path(__file__).resolve().parents[3] / "third_party" / "DCVC" / "DCVC-family" / "DCVC-DC"
DEFAULT_CKPT_I = str(_DCVC_DC_DIR / "checkpoints" / "cvpr2023_image_psnr.pth.tar")
DEFAULT_CKPT_P = str(_DCVC_DC_DIR / "checkpoints" / "cvpr2023_video_psnr.pth.tar")


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


@register_codec("dcvc_dc")
class DcvcDcCodec(VideoCodec):
    """DCVC-DC (CVPR 2023) — IntraNoAR I-frame + DMC P-frame, MLCodec_rans ext."""

    name = "dcvc_dc"
    family = "dcvc_dc"
    ext = "bin"
    is_neural = True
    browser_playable = False
    _CACHE: dict = {}

    def __init__(self, crf: int = 1, preset: str | None = None,
                 checkpoint_i: str | None = None, checkpoint_p: str | None = None):
        super().__init__(crf=crf, preset=preset)
        self.q_index = int(crf)
        self.checkpoint_i = checkpoint_i
        self.checkpoint_p = checkpoint_p
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = None

    def _setup_error(self, reason: str) -> RuntimeError:
        return RuntimeError(
            f"DCVC-DC setup incomplete: {reason}\n"
            "  1. src: third_party/DCVC/DCVC-family/DCVC-DC/src/ (git submodule)\n"
            "  2. checkpoints: download cvpr2023_image_psnr.pth.tar + cvpr2023_video_psnr.pth.tar\n"
            f"     from OneDrive (see t10-1 in tasks.json for URLs) -> {_DCVC_DC_DIR}/checkpoints/\n"
            "  3. rans ext: MLCodec_rans + MLCodec_CXX .pyd in src/models/ (built from src/cpp, cp312)\n"
            f"     I ckpt: {self.checkpoint_i or DEFAULT_CKPT_I}\n"
            f"     P ckpt: {self.checkpoint_p or DEFAULT_CKPT_P}"
        )

    def _load(self) -> dict:
        key = (self.device, self.checkpoint_i, self.checkpoint_p)
        if key in DcvcDcCodec._CACHE:
            return DcvcDcCodec._CACHE[key]
        repo = str(_DCVC_DC_DIR)
        if not os.path.isdir(os.path.join(repo, "src", "models")):
            raise self._setup_error("DCVC-DC src not found")
        if repo not in sys.path:
            sys.path.insert(0, repo)

        try:
            from src.models.MLCodec_rans import RansEncoder  # noqa: F401
            from src.models.MLCodec_CXX import pmf_to_quantized_cdf  # noqa: F401
        except ImportError as exc:
            raise self._setup_error(
                f"MLCodec_rans/MLCodec_CXX ext not importable ({exc}); "
                "build from src/cpp with cmake (cp312)"
            ) from exc

        from src.models.image_model import IntraNoAR
        from src.models.video_model import DMC
        from src.utils.stream_helper import (
            get_state_dict,
            get_padding_size,
            decode_i,
            decode_p,
        )

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

        bundle = {
            "i_net": i_net, "p_net": p_net,
            "get_padding_size": get_padding_size,
            "decode_i": decode_i, "decode_p": decode_p,
        }
        DcvcDcCodec._CACHE[key] = bundle
        return bundle

    @property
    def models(self) -> dict:
        if self._models is None:
            self._models = self._load()
        return self._models

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        bundle = self.models
        i_net, p_net = bundle["i_net"], bundle["p_net"]
        get_padding_size = bundle["get_padding_size"]
        h0, w0 = frames[0].shape[:2]
        pad_l, pad_r, pad_t, pad_b = get_padding_size(h0, w0, 16)

        records: list[tuple[int, int, bytes]] = []
        stats_list: list[dict] = []
        tmp_dir = tempfile.mkdtemp(prefix="dcvc_dc_enc_")
        try:
            with torch.no_grad():
                x0, st0 = _img_to_tensor(frames[0])
                x0 = x0.to(self.device)
                x0_pad = torch.nn.functional.pad(x0, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
                bin_path = os.path.join(tmp_dir, "f000000.bin")
                result = i_net.encode_decode(x0_pad, True, self.q_index, bin_path,
                                            pic_height=h0, pic_width=w0)
                dpb = _seed_dpb(result["x_hat"])
                with open(bin_path, "rb") as f:
                    records.append((0, self.q_index, f.read()))
                stats_list.append(st0)

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
                    bin_path = os.path.join(tmp_dir, f"f{i:06d}.bin")
                    result = p_net.encode_decode(xi_pad, dpb, True, self.q_index,
                                                x_next_pad, bin_path,
                                                pic_height=h0, pic_width=w0, frame_idx=i % 4)
                    dpb = result["dpb"]
                    with open(bin_path, "rb") as f:
                        records.append((1, self.q_index, f.read()))
                    stats_list.append(sti)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return self._serialize(records, len(frames), h0, w0, stats_list)

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int, hw: tuple[int, int]) -> list:
        bundle = self.models
        i_net, p_net = bundle["i_net"], bundle["p_net"]
        decode_i, decode_p = bundle["decode_i"], bundle["decode_p"]
        records, (h0, w0), stats_list = self._deserialize(bitstream_bytes)
        if len(records) != n_frames:
            raise RuntimeError(
                f"DCVC-DC bitstream frame count {len(records)} != requested {n_frames}"
            )

        tmp_dir = tempfile.mkdtemp(prefix="dcvc_dc_dec_")
        recons: list[np.ndarray] = []
        try:
            with torch.no_grad():
                for idx, (ftype, _q, bs) in enumerate(records):
                    bin_path = os.path.join(tmp_dir, f"f{idx:06d}.bin")
                    with open(bin_path, "wb") as f:
                        f.write(bs)
                    if ftype == 0:
                        h, w, q, q_idx, string = decode_i(bin_path)
                        dec = i_net.decompress(string, h, w, q, q_idx)
                        x_hat = dec["x_hat"]
                        dpb = _seed_dpb(x_hat)
                    else:
                        q, q_idx, fidx, string = decode_p(bin_path)
                        dec = p_net.decompress(dpb, string, h0, w0, q, q_idx, fidx)
                        dpb = dec["dpb"]
                        x_hat = dpb["ref_frame"]
                    x_hat = x_hat[:, :, :h0, :w0].float()
                    recons.append(_tensor_to_img(x_hat, stats_list[idx]))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return recons

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
            raise RuntimeError(f"DCVC-DC bitstream magic mismatch (got {buf[:8]!r})")
        off = len(_MAGIC)
        n, h, w = _HDR.unpack_from(buf, off); off += _HDR.size
        records = []
        for _ in range(n):
            ftype, q, blen = _FREC.unpack_from(buf, off); off += _FREC.size
            records.append((ftype, q, buf[off:off + blen])); off += blen
        (slen,) = _LEN.unpack_from(buf, off); off += _LEN.size
        stats_list = pickle.loads(buf[off:off + slen])
        return records, (h, w), stats_list
