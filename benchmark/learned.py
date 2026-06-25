"""Image compression benchmark — learned compression models via CompressAI."""

import warnings
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import CompressionResult, compute_psnr, compute_ssim, timed


# CompressAI model names and their quality levels (1-8)
LEARNED_MODELS = [
    ("bmshj2018-factorized", [1, 4, 8]),   # Factorized Prior
    ("bmshj2018-hyperprior", [1, 4, 8]),   # Hyperprior
    ("mbt2018-mean", [1, 4, 8]),           # Mean Scale Hyperprior
    ("mbt2018", [1, 4, 8]),                # Scale Hyperprior
    ("cheng2020-anchor", [1, 4, 6]),       # Channel Autoregressive (q1-q6 only)
]

# Model cache: avoids reloading for every image
_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_name: str, quality: int, device: str = "cpu"):
    """Load a pretrained CompressAI model (cached).

    If the model weights are not already downloaded, raises RuntimeError
    instead of attempting a slow download.
    """
    cache_key = f"{model_name}_{quality}_{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from compressai.zoo import image_models
    from compressai.zoo.image import model_urls
    import os

    # Pre-check: verify model weights are cached before attempting download
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

    model_cls = image_models[model_name]
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


def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert numpy image to (1, 3, H, W) float tensor in [0, 1]."""
    if img.dtype == np.uint16:
        arr = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        arr = img.astype(np.float32) / 255.0
    else:
        arr = img.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)  # (3, H, W)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
    else:
        arr = np.stack([arr, arr, arr], axis=0)

    return torch.from_numpy(arr).unsqueeze(0)  # (1, 3, H, W)


def _tensor_to_img(t: torch.Tensor, ref_dtype: np.dtype) -> np.ndarray:
    """Convert (1, 3, H, W) tensor back to numpy, taking first channel."""
    arr = t.squeeze(0).cpu().numpy()  # (3, H, W)
    arr = arr[0]  # take first channel (grayscale)
    arr = np.clip(arr, 0, 1)
    if ref_dtype == np.uint8:
        return (arr * 255).astype(np.uint8)
    elif ref_dtype == np.uint16:
        return (arr * 65535).astype(np.uint16)
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
    x = _img_to_tensor(img).to(device)

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
    reconstructed = _tensor_to_img(x_hat, img.dtype)

    # Align for metrics
    ref = img
    rec_for_metrics = reconstructed
    if ref.dtype == np.uint16:
        ref = ref.astype(np.float32) / 65535.0
        rec_for_metrics = reconstructed.astype(np.float32) / 65535.0
    elif ref.dtype == np.uint8:
        ref = ref.astype(np.float32) / 255.0
        rec_for_metrics = reconstructed.astype(np.float32) / 255.0

    psnr = compute_psnr(ref, rec_for_metrics)
    ssim = compute_ssim(ref, rec_for_metrics)
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
