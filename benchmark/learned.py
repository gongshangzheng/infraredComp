"""Image compression benchmark — learned compression models via CompressAI."""

import warnings
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import CompressionResult, compute_psnr, compute_ssim, timed


# ELIC quality levels: 1=low bpp (~0.1), 4=medium (~0.5), 5=high (~1.0)
ELIC_QUALITIES = [1, 4, 5]

# CompressAI model names and their quality levels (1-8)
LEARNED_MODELS = [
    ("bmshj2018-factorized", [1, 4, 8]),   # Factorized Prior
    ("bmshj2018-hyperprior", [1, 4, 8]),   # Hyperprior
    ("mbt2018-mean", [1, 4, 8]),           # Mean Scale Hyperprior
    ("mbt2018", [1, 4, 8]),                # Scale Hyperprior
    ("cheng2020-anchor", [1, 4, 6]),       # Channel Autoregressive (q1-q6 only)
    ("cheng2020-attn", [1, 4, 6]),           # Attention-guided (q1-q6 only)
]

# Model cache: avoids reloading for every image
_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_name: str, quality: int, device: str = "cpu", checkpoint_path: str | None = None):
    """Load a CompressAI model (cached).

    If ``checkpoint_path`` is given, instantiate fresh (pretrained=False) and load the
    trained state_dict — this is the checkpoint→eval hook (use a model trained via
    scripts/train_model.py for evaluation). Otherwise load pretrained (cached).
    """
    cache_key = f"{model_name}_{quality}_{device}_{checkpoint_path or 'pretrained'}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from compressai.zoo import image_models
    import os

    model_cls = image_models[model_name]
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"Trained checkpoint not found: {checkpoint_path}")
        model = model_cls(quality=quality, pretrained=False)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    else:
        from compressai.zoo.image import model_urls
        # Pre-check: verify pretrained weights are cached before attempting download
        try:
            url = model_urls[model_name]["mse"][quality]
            fname = url.rsplit("/", 1)[-1]
            cache_path = os.path.expanduser(f"~/.cache/torch/hub/checkpoints/{fname}")
            if not os.path.isfile(cache_path):
                raise RuntimeError(
                    f"Model weights not cached: {fname}. "
                    f"Download from {url} first."
                )
        except (KeyError, TypeError):
            pass  # URL not in registry; let CompressAI handle the error
        model = model_cls(quality=quality, pretrained=True)
    model = model.to(device)
    model.eval()
    model.update()
    _MODEL_CACHE[cache_key] = model
    return model


def _pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad tensor to be divisible by `multiple`. Returns (padded, (pad_h, pad_w))."""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def _unpad(x: torch.Tensor, pad: tuple[int, int]) -> torch.Tensor:
    """Remove padding from tensor."""
    pad_h, pad_w = pad
    if pad_h or pad_w:
        x = x[:, :, : x.shape[2] - pad_h, : x.shape[3] - pad_w]
    return x


def _img_to_tensor(img: np.ndarray) -> tuple[torch.Tensor, dict]:
    """Convert numpy image to (1, 3, H, W) float tensor in [0, 1].

    For uint16 / uint8 inputs, applies min-max normalization so that the signal
    fills [0, 1]. This matches the approach used by traditional codecs
    (which also stretch the signal before compression) and ensures PSNR values
    are comparable across codec families.

    Returns (tensor, norm_stats) where norm_stats is needed for denormalization.
    """
    stats: dict = {}

    if img.dtype == np.uint16:
        img_min, img_max = int(img.min()), int(img.max())
        if img_max > img_min:
            arr = (img.astype(np.float32) - img_min) / (img_max - img_min)
        else:
            arr = np.zeros_like(img, dtype=np.float32)
        stats = {"src_dtype": "uint16", "img_min": img_min, "img_max": img_max}
    elif img.dtype == np.uint8:
        img_min, img_max = int(img.min()), int(img.max())
        if img_max > img_min:
            arr = (img.astype(np.float32) - img_min) / (img_max - img_min)
        else:
            arr = np.zeros_like(img, dtype=np.float32)
        stats = {"src_dtype": "uint8", "img_min": img_min, "img_max": img_max}
    else:
        arr = img.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        stats = {"src_dtype": "float", "img_min": float(arr_min), "img_max": float(arr_max)}

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
    else:
        arr = np.stack([arr, arr, arr], axis=0)

    return torch.from_numpy(arr).unsqueeze(0), stats  # (1, 3, H, W)


def _tensor_to_img(
    t: torch.Tensor,
    norm_stats: dict,
) -> np.ndarray:
    """Convert (1, 3, H, W) tensor back to numpy, reversing the normalization
    applied by ``_img_to_tensor``.

    Parameters
    ----------
    t : tensor in [0, 1] (min-max normalized)
    norm_stats : dict returned by ``_img_to_tensor``
    """
    arr = t.squeeze(0).cpu().numpy()  # (3, H, W)
    arr = arr[0]  # take first channel (grayscale)
    arr = np.clip(arr, 0, 1)

    src_dtype = norm_stats.get("src_dtype", "float")
    img_min = norm_stats.get("img_min", 0)
    img_max = norm_stats.get("img_max", 1)

    if src_dtype == "uint8":
        return (arr * (img_max - img_min) + img_min).astype(np.uint8)
    elif src_dtype == "uint16":
        return (arr * (img_max - img_min) + img_min).astype(np.uint16)
    else:
        return arr.astype(np.float32)


def compress_learned(
    img: np.ndarray,
    name: str,
    model_name: str,
    quality: int,
    device: str = "cpu",
) -> CompressionResult:
    """Compress an image with a CompressAI neural codec."""
    model = _load_model(model_name, quality, device)
    x, norm_stats = _img_to_tensor(img)
    x = x.to(device)

    # Pad to multiple of 64 (required by most learned codecs)
    x_padded, pad = _pad_to_multiple(x, 64)
    _, _, orig_h, orig_w = x.shape

    with torch.no_grad():

        def _encode():
            return model.compress(x_padded)

        out_enc, encode_ms = timed(_encode)

        # Bitstream size
        bits = sum(len(strings[0]) * 8 for strings in out_enc["strings"])
        compressed_bytes = bits // 8

        def _decode():
            return model.decompress(out_enc["strings"], out_enc["shape"])

        out_dec, decode_ms = timed(_decode)

    # Unpad
    x_hat = _unpad(out_dec["x_hat"], pad)

    # Reconstruct in original dtype using the stored normalization stats
    reconstructed = _tensor_to_img(x_hat, norm_stats)

    # Metric computation on float [0,1] min-max normalized arrays.
    # This is equivalent to how traditional codecs compute PSNR after stretching
    # the signal to fill [0, 255]: both measure quality on the stretched signal,
    # making the PSNR values directly comparable.
    ref_norm = x.squeeze(0)[0].cpu().numpy()  # (H, W) in [0,1] first channel
    rec_norm = x_hat.squeeze(0)[0].cpu().numpy()  # (H, W) in [0,1] first channel
    rec_norm = np.clip(rec_norm, 0, 1)

    psnr = compute_psnr(ref_norm, rec_norm)
    ssim = compute_ssim(ref_norm, rec_norm)
    pixel_count = orig_h * orig_w
    bpp = bits / pixel_count
    compression_ratio = img.nbytes / compressed_bytes if compressed_bytes else 0

    return CompressionResult(
        codec=f"{model_name}-q{quality}",
        image_name=name,
        psnr=psnr,
        ssim=ssim,
        bpp=bpp,
        compression_ratio=compression_ratio,
        encode_time_ms=encode_ms,
        decode_time_ms=decode_ms,
        original_shape=img.shape,
        compressed_bytes=compressed_bytes,
        decoded=reconstructed,
    )


def get_learned_codecs(device: str = "cpu") -> list[Callable]:
    """Return list of learned codec functions for all models and quality levels."""
    codecs = []
    for model_name, qualities in LEARNED_MODELS:
        for q in qualities:

            def make_codec(m=model_name, qual=q):
                def codec(img: np.ndarray, name: str) -> CompressionResult:
                    return compress_learned(img, name, m, qual, device)

                return codec

            codecs.append(make_codec())
    return codecs


def _count_elic_bits(out_enc: dict) -> int:
    """Count total bits in ELIC's non-standard bitstream format.

    ELIC compress returns:
        strings[0]: list of [anchor_strings, non_anchor_strings] per slice
        strings[1]: list of bytes for z (hyperprior)
    Each stream element is either bytes or list[bytes] (batch dim).
    """
    bits = 0
    # y strings: list of [anchor_strings, non_anchor_strings] per slice
    for slice_pair in out_enc["strings"][0]:
        for stream in slice_pair:
            if isinstance(stream, list):
                for b in stream:
                    if isinstance(b, bytes):
                        bits += len(b) * 8
            elif isinstance(stream, bytes):
                bits += len(stream) * 8
    # z strings: list of bytes
    for b in out_enc["strings"][1]:
        if isinstance(b, bytes):
            bits += len(b) * 8
        elif isinstance(b, list):
            for bb in b:
                bits += len(bb) * 8
    return bits


def _load_elic_model(quality: int, device: str = "cpu"):
    """Load a pretrained ELIC model (cached)."""
    cache_key = f"elic_{quality}_{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from .elic_model import load_elic_model
    model = load_elic_model(quality, device)
    _MODEL_CACHE[cache_key] = model
    return model


def compress_elic(
    img: np.ndarray,
    name: str,
    quality: int,
    device: str = "cpu",
) -> CompressionResult:
    """Compress an image with ELIC (Efficient Learned Image Compression)."""
    model = _load_elic_model(quality, device)
    x, norm_stats = _img_to_tensor(img)
    x = x.to(device)

    # ELIC uses 64x downsampling (4 conv layers with stride 2 + 2 hyper layers)
    x_padded, pad = _pad_to_multiple(x, 64)
    _, _, orig_h, orig_w = x.shape

    with torch.no_grad():
        def _encode():
            return model.compress(x_padded)

        out_enc, encode_ms = timed(_encode)
        bits = _count_elic_bits(out_enc)
        compressed_bytes = bits // 8

        def _decode():
            return model.decompress(out_enc["strings"], out_enc["shape"])

        out_dec, decode_ms = timed(_decode)

    x_hat = _unpad(out_dec["x_hat"], pad)
    reconstructed = _tensor_to_img(x_hat, norm_stats)

    ref_norm = x.squeeze(0)[0].cpu().numpy()
    rec_norm = x_hat.squeeze(0)[0].cpu().numpy()
    rec_norm = np.clip(rec_norm, 0, 1)

    psnr = compute_psnr(ref_norm, rec_norm)
    ssim = compute_ssim(ref_norm, rec_norm)
    pixel_count = orig_h * orig_w
    bpp = bits / pixel_count
    compression_ratio = img.nbytes / compressed_bytes if compressed_bytes else 0

    return CompressionResult(
        codec=f"ELIC-q{quality}",
        image_name=name,
        psnr=psnr,
        ssim=ssim,
        bpp=bpp,
        compression_ratio=compression_ratio,
        encode_time_ms=encode_ms,
        decode_time_ms=decode_ms,
        original_shape=img.shape,
        compressed_bytes=compressed_bytes,
        decoded=reconstructed,
    )


def get_elic_codecs(device: str = "cpu") -> list[Callable]:
    """Return list of ELIC codec functions for all quality levels."""
    codecs = []
    for q in ELIC_QUALITIES:
        def make_codec(qual=q):
            def codec(img: np.ndarray, name: str) -> CompressionResult:
                return compress_elic(img, name, qual, device)
            return codec
        codecs.append(make_codec())
    return codecs
