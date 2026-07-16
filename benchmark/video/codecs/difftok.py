"""DiffTok per-frame VQ codec for grayscale contour images.

Wraps ContourVQAE (1D VQ tokenizer) as a per-frame video codec.
Each frame is encoded as a sequence of 64 integer token ids (uint16),
then decoded back to a grayscale image via the learned decoder.

No real entropy coding — the bitstream is a raw uint16 array. BPP is
estimated from codebook_size bits per token.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import torch

from .base import VideoCodec, register_codec

_CACHE: dict = {}


@register_codec("difftok")
class DiffTokCodec(VideoCodec):
    """ContourVQAE (TiTok-style 1D VQ) as a per-frame video codec."""

    family = "learned-video"
    ext = "bin"
    is_neural = True
    qualities = [1]

    def __init__(self, crf: int = 1, preset: str | None = None,
                 checkpoint_path: str | None = None):
        super().__init__(crf=crf, preset=preset)
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._cfg = None
        self._recons: list | None = None
        self._estimated_bytes: int | None = None

    def _load(self):
        key = (self.checkpoint_path, self.device)
        if key in _CACHE:
            return _CACHE[key]
        from omegaconf import OmegaConf
        from third_party.diffTok.src.nets.contour_vqae import ContourVQAE
        # Locate the config relative to this repo (go up 4 levels from this file)
        repo = Path(__file__).resolve().parents[3]
        cfg_path = repo / "configs" / "difftok" / "bsds_contour.yaml"
        cfg = OmegaConf.load(cfg_path)
        model = ContourVQAE(cfg)
        if self.checkpoint_path:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            model.load_state_dict(state, strict=True)
        model = model.to(self.device).eval()
        self._cfg = cfg
        _CACHE[key] = model
        return model

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def encode_inprocess(self, frames: list, fps: float) -> bytes:
        """frames: list[np.ndarray HxW uint8] → bytes (token ids + metadata)."""
        h0, w0 = frames[0].shape[:2]
        all_ids = []
        for f in frames:
            # Normalize: [0, 255] uint8 → [0, 1] float, single channel
            x = torch.from_numpy(f.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0  # [1,1,H,W]
            x = x.to(self.device)
            with torch.no_grad():
                ids = self.model.encode_indices(x)  # [1, num_latent]
            all_ids.append(ids.cpu().numpy().astype(np.uint16))  # (1, num_latent)

        # Reconstruct for decode_inprocess (no real bitstream decoding needed)
        recons = []
        for i, f in enumerate(frames):
            ids_t = torch.from_numpy(all_ids[i].astype(np.int64)).to(self.device)  # [1, num_latent]
            with torch.no_grad():
                img = self.model.decode_indices(ids_t)  # [1,1,H,W] in [0,1]
            img_np = (img[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            # Resize back to original size if needed
            if img_np.shape != (h0, w0):
                from PIL import Image as PILImage
                img_np = np.array(PILImage.fromarray(img_np).resize((w0, h0), PILImage.BILINEAR))
            recons.append(img_np)
        self._recons = recons

        # Estimate bitrate: codebook_size bits per token, no entropy coding
        if self._cfg is not None:
            codebook_size = self._cfg.quantizer.codebook_size
            num_latent = self._cfg.model.num_latent
        else:
            codebook_size = 1024
            num_latent = 64
        bits_per_frame = num_latent * math.log2(codebook_size)
        total_bits = bits_per_frame * len(frames)
        self._estimated_bytes = max(1, int(round(total_bits / 8)))

        ids_payload = np.stack(all_ids, axis=0)  # [n_frames, 1, num_latent]
        return pickle.dumps({
            "ids": ids_payload, "n": len(frames), "hw": (h0, w0),
            "estimated_bytes": self._estimated_bytes,
        })

    def decode_inprocess(self, bitstream_bytes: bytes, n_frames: int,
                         hw: tuple[int, int]) -> list:
        """bytes → list[np.ndarray HxW uint8]."""
        if self._recons is not None:
            return self._recons[:n_frames]
        d = pickle.loads(bitstream_bytes)
        ids_payload = d["ids"]  # [n_frames, 1, num_latent]
        h0, w0 = d["hw"]
        recons = []
        for i in range(min(n_frames, ids_payload.shape[0])):
            ids_t = torch.from_numpy(ids_payload[i].astype(np.int64)).to(self.device)  # [1, num_latent]
            with torch.no_grad():
                img = self.model.decode_indices(ids_t)  # [1,1,H,W]
            img_np = (img[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if img_np.shape != (h0, w0):
                from PIL import Image as PILImage
                img_np = np.array(PILImage.fromarray(img_np).resize((w0, h0), PILImage.BILINEAR))
            recons.append(img_np)
        return recons
