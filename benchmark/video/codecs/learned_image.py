"""Per-frame learned video codec — wraps CompressAI IMAGE models as video codecs.

把 CompressAI 图像模型（bmshj2018/cheng2020/ELIC）当 per-frame 视频 codec：把轮廓帧序列
的**每一帧独立**压缩/解压（无时序建模，每帧当一张图）。比 ssf2020 简单——无 keyframe/inter，
直接复用 `benchmark/learned.py` 的单图 compress/decompress 模式。

通道/范围转换（用户强调）：每帧 `_img_to_tensor`（gray→3ch + min-max norm 到 [0,1]）→
model.compress → model.decompress → `_tensor_to_img`（denorm 回 uint8 单通道）。stats + pad
序列化进 bitstream，decode 无原始帧也能反 norm。pretrained 在自然图像训练，轮廓 OOD → 可 fine-tune。
"""
from __future__ import annotations

import pickle

import torch

from .base import VideoCodec, register_codec
from benchmark.learned import (
    _img_to_tensor, _tensor_to_img, _pad_to_multiple, _unpad, _load_model,
)

# 图像模型 + 各自 quality 级（与 evaluation.py _DL_MODELS 一致）
_IMG_MODELS = [
    ("bmshj2018-factorized", [1, 4, 8]),
    ("bmshj2018-hyperprior", [1, 4, 8]),
    ("mbt2018-mean", [1, 4, 8]),
    ("mbt2018", [1, 4, 8]),
    ("cheng2020-anchor", [1, 4, 6]),
    ("cheng2020-attn", [1, 4, 6]),
    ("ELIC", [1, 4, 5]),
]


class _LearnedImageVideoCodec(VideoCodec):
    """Per-frame CompressAI image model as a video codec (no temporal)."""

    family = "learned-video"
    ext = "bin"
    is_neural = True
    model_name: str = ""          # set per-registration
    qualities: list = []          # set per-registration
    _CACHE: dict = {}

    def __init__(self, crf: int, preset: str | None = None, checkpoint_path: str | None = None,
                 estimate: bool = True):
        super().__init__(crf=crf, preset=preset)
        self.quality = crf
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        # estimate=True(默认):走 model(x) forward 拿 x_hat+likelihoods,bpp 用 -log2(lik) 估计,
        # 不跑 rans 范围编码 → 纯 GPU、不 segfault、快。代价:bpp 是估计值(误差<1%),无真实 .bin。
        # estimate=False:走 model.compress/decompress(真实 rans bitstream,慢 + 累积 segfault)。
        self.estimate = estimate
        self._recons = None            # estimate 模式下缓存重建帧供 decode 用
        self._estimated_bytes = None   # estimate 模式下报告的压缩字节数(estimated_bits/8)

    def _load(self):
        key = (self.model_name, self.quality, self.device, self.checkpoint_path)
        if key in _LearnedImageVideoCodec._CACHE:
            return _LearnedImageVideoCodec._CACHE[key]
        if self.model_name == "ELIC":
            from benchmark.elic_model import load_elic_model
            m = load_elic_model(self.quality, self.device, self.checkpoint_path)
        else:
            m = _load_model(self.model_name, self.quality, self.device, self.checkpoint_path)
        _LearnedImageVideoCodec._CACHE[key] = m
        return m

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    # ---- per-frame in-process encode/decode ----

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        """frames = list[np.ndarray HxW uint8] -> bytes。

        estimate=True: model(x) forward(GPU) 拿 x_hat + likelihoods;不跑 rans。
          bpp = Σ -log2(likelihood) / 像素数(标准 RD 估计,论文通用)。
          重建帧缓存到 self._recons 供 decode 返回(无真实 bitstream 可解码)。
          返回极小 pickle(仅 n/hw/estimated_bytes);compressed_bytes 走 self._estimated_bytes。
        estimate=False: model.compress 跑真实 rans(慢 + 累积 segfault)。"""
        h0, w0 = frames[0].shape[:2]
        if not self.estimate:
            return self._encode_rans(frames, h0, w0)
        import math
        recons = []
        total_bits = 0.0
        for f in frames:
            x, stats = _img_to_tensor(f)
            x, pad = _pad_to_multiple(x, 64)
            x = x.to(self.device)
            with torch.no_grad():
                out = self.model(x)               # {"x_hat", "likelihoods"}
            liks = out["likelihoods"]
            if isinstance(liks, dict):             # 命名 likelihoods dict -> 取 values
                liks = liks.values()
            for lik in liks:
                total_bits += float((-torch.log2(lik.clamp(min=1e-9))).sum().item())
            x_hat = _unpad(out["x_hat"], pad)
            recons.append(_tensor_to_img(x_hat, stats))   # HxW uint8
        self._recons = recons
        self._estimated_bytes = max(1, int(round(total_bits / 8)))
        # 极小 bitstream(仅元数据);真实 compressed_bytes 由 self._estimated_bytes 报告。
        return pickle.dumps({"n": len(recons), "hw": (h0, w0),
                              "estimated_bytes": self._estimated_bytes, "estimate": True})

    def _encode_rans(self, frames, h0, w0):
        """真实 rans 路径(estimate=False):model.compress 产 bitstream + model.decompress 出重建。"""
        per_frame = []
        for f in frames:
            x, stats = _img_to_tensor(f)
            x, pad = _pad_to_multiple(x, 64)
            x = x.to(self.device)
            with torch.no_grad():
                out = self.model.compress(x)              # {"strings", "shape"}
            per_frame.append({"strings": out["strings"], "shape": out["shape"], "stats": stats, "pad": pad})
        return pickle.dumps({"frames": per_frame, "n": len(per_frame), "hw": (h0, w0), "estimate": False})

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int, hw: tuple[int, int]) -> list:
        """bytes -> list[np.ndarray HxW uint8]。

        estimate 模式:无真实 bitstream,直接返回 encode 时缓存的 self._recons(forward 的 x_hat)。
        rans 模式:model.decompress(strings, shape) 解码。"""
        d = pickle.loads(bitstream_bytes)
        if d.get("estimate") or self._recons is not None:
            return self._recons[:n_frames] if self._recons else []
        recons = []
        for fr in d["frames"]:
            with torch.no_grad():
                out = self.model.decompress(fr["strings"], fr["shape"])
            x_hat = _unpad(out["x_hat"], fr["pad"])
            recons.append(_tensor_to_img(x_hat, fr["stats"]))
        return recons


# 动态注册每个图像模型为一个 video codec: img-<model_name>
for _name, _quals in _IMG_MODELS:
    _cls = type(
        f"LearnedImageVideoCodec_{_name.replace('-', '_')}",
        (_LearnedImageVideoCodec,),
        {"model_name": _name, "qualities": _quals, "name": f"img-{_name}"},
    )
    register_codec(f"img-{_name}")(_cls)
