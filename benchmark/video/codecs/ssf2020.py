"""CompressAI ssf2020 (Scale-Space Flow) learned video codec.

In-process neural codec (no ffmpeg): loads `compressai.zoo.video_models["ssf2020"]`,
encodes a contour frame sequence via `encode_keyframe` + `encode_inter` loop,
serializes strings+shapes to bytes (pickle); decodes via `decode_keyframe`/`decode_inter`.

`crf` = ssf2020 quality (1-9). `checkpoint_path` (optional) overrides pretrained weights
(checkpoint→eval hook: use a model fine-tuned on contour data via scripts/train_model.py).

Reuses `benchmark/learned.py` helpers: `_img_to_tensor`/`_tensor_to_img` (min-max norm +
gray→3ch), `_pad_to_multiple`/`_unpad` (÷64). See `.claude/skills/compressai-usage`.
"""
from __future__ import annotations

import pickle

import numpy as np
import torch

from .base import VideoCodec, register_codec
from .. import config  # noqa: F401  (keeps parity with legacy codecs re: config import)
from benchmark.learned import _img_to_tensor, _tensor_to_img, _pad_to_multiple, _unpad


@register_codec("ssf2020")
class SSF2020Codec(VideoCodec):
    """CompressAI ssf2020 learned video codec (in-process, neural)."""

    name = "ssf2020"
    family = "learned-video"
    ext = "bin"            # bitstream = serialized bytes (not a media container)
    is_neural = True

    _CACHE: dict = {}

    def __init__(self, crf: int, preset: str | None = None, checkpoint_path: str | None = None):
        super().__init__(crf=crf, preset=preset)
        # ssf2020 quality 1-9;若传入 CRF(>9,来自 --crfs 统一接口),反向映射
        # (CRF 高=质量低 -> quality 低;CRF 低=质量高 -> quality 高)
        self.quality = crf if 1 <= crf <= 9 else max(1, min(9, 10 - crf // 4))
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None            # lazy-load on first encode/decode

    def _load(self):
        key = (self.quality, self.device, self.checkpoint_path)
        if key in SSF2020Codec._CACHE:
            return SSF2020Codec._CACHE[key]
        from compressai.zoo import video_models
        if self.checkpoint_path:
            m = video_models["ssf2020"](quality=self.quality, metric="mse", pretrained=False)
            m.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=False))
        else:
            m = video_models["ssf2020"](quality=self.quality, metric="mse", pretrained=True)
        m = m.to(self.device).eval()
        m.update()                    # required before compress/decompress
        SSF2020Codec._CACHE[key] = m
        return m

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    # ---- in-process encode/decode ----

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        """frames = list[np.ndarray HxW uint8] -> bytes (pickled strings+shapes)."""
        tframes, stats_list, pads = [], [], []
        h0, w0 = frames[0].shape[:2]
        for f in frames:
            t, st = _img_to_tensor(f)                 # (1,3,H,W) float [0,1] + norm_stats
            t, pad = _pad_to_multiple(t, 64)
            tframes.append(t.to(self.device)); stats_list.append(st); pads.append(pad)
        with torch.no_grad():
            x_hat_kf, out_kf = self.model.encode_keyframe(tframes[0])
            strings = [out_kf["strings"]]; shapes = [out_kf["shape"]]
            x_ref = x_hat_kf
            for i in range(1, len(tframes)):
                x_ref, out_i = self.model.encode_inter(tframes[i], x_ref)
                strings.append(out_i["strings"]); shapes.append(out_i["shape"])
        return pickle.dumps({
            "strings": strings, "shapes": shapes,
            "n": len(tframes), "stats": stats_list, "pads": pads, "hw": (h0, w0),
        })

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int, hw: tuple[int, int]) -> list:
        """bytes -> list[np.ndarray HxW uint8] (decoded contour frames)."""
        d = pickle.loads(bitstream_bytes)
        strings, shapes = d["strings"], d["shapes"]
        stats_list, pads = d["stats"], d["pads"]
        recons: list[np.ndarray] = []
        with torch.no_grad():
            x_ref = self.model.decode_keyframe(strings[0], shapes[0])
            recons.append(_tensor_to_img(_unpad(x_ref, pads[0]), stats_list[0]))
            for i in range(1, len(strings)):
                x_ref = self.model.decode_inter(x_ref, strings[i], shapes[i])
                recons.append(_tensor_to_img(_unpad(x_ref, pads[i]), stats_list[i]))
        return recons
