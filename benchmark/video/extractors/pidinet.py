"""PiDiNet (Pixel Difference Network) extractor — deep edge detector.

Vendors `hellozhuo/pidinet` model code under ``third_party/pidinet/models/`` and
runs the **converted** (vanilla-conv, re-parameterized) PiDiNet — config
``carv4`` with CSAM (``sa``) + CDCM (``dil``), the BSDS release
(``table5_pidinet.pth``, committed in the upstream repo, ~3 MB). Same interface
as canny/sobel/hed: ``extract(frame) -> uint8 HxW``.

This is the first **torch-based** extractor in this package (HED uses cv2.dnn).
The torch load idiom mirrors the learned codecs (``codecs/ssf2020.py``,
``codecs/learned_image.py``): pick cuda if available, ``weights_only=False``,
``.eval()``, a module-level ``_CACHE`` keyed by device so repeated
``build_extractor`` calls don't reload.

Preprocessing (from PiDiNet ``edge_dataloader.py``): grayscale → 3-channel RGB →
``ToTensor`` → ImageNet ``Normalize`` → ``unsqueeze(0)``. PiDiNet is fully
convolutional (no fixed input size, no padding). ``forward`` returns a list of
5 sigmoid'd side outputs; ``[-1]`` is the fused map already bilinearly resized
back to the input HxW, so no extra sigmoid/resize is needed — just ×255.

Weights must be vendored at ``third_party/pidinet/table5_pidinet.pth`` (run
``scripts/download_pidinet_weights.py``); this module pre-checks and raises a
clear error if missing, mirroring the HED guard.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .base import ContourExtractor, register

# Vendored model code + weight under third_party/pidinet/.
# pidinet.py = benchmark/video/extractors/pidinet.py -> parents[3] = repo root.
_PIDINET_DIR = Path(__file__).resolve().parents[3] / "third_party" / "pidinet"
DEFAULT_WEIGHTS = str(_PIDINET_DIR / "table5_pidinet.pth")

# ImageNet mean/std (PiDiNet edge_dataloader.py: ToTensor then Normalize).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# device -> (model, transform); cached so build_extractor doesn't reload the .pth.
_CACHE: dict[str, tuple] = {}


def _load(device: str):
    """Build the converted PiDiNet (carv4, sa, dil) with re-parameterized
    weights, eval mode. Cached per device."""
    if device in _CACHE:
        return _CACHE[device]
    import torch
    from torchvision import transforms

    # Make vendored models/ importable. (Mirrors dcvc_rt's sys.path approach.)
    if str(_PIDINET_DIR) not in sys.path:
        sys.path.insert(0, str(_PIDINET_DIR))
    import models  # noqa: E402  (third_party/pidinet/models)
    from models.convert_pidinet import convert_pidinet  # noqa: E402

    args = types.SimpleNamespace(config="carv4", sa=True, dil=True)
    net = models.pidinet_converted(args)  # vanilla-conv, inplane=60, dil=24, sa
    ckpt = torch.load(DEFAULT_WEIGHTS, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    # checkpoint saved under torch.nn.DataParallel -> keys prefixed "module."; we
    # instantiate a single (non-DataParallel) model -> strip the prefix so keys match.
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    # checkpoint was saved for the PDC model; fold PDC weights into vanilla convs.
    net.load_state_dict(convert_pidinet(sd, "carv4"))
    net = net.to(device).eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    _CACHE[device] = (net, transform)
    return _CACHE[device]


@register("pidinet")
class PidinetExtractor(ContourExtractor):
    """PiDiNet (carv4, converted) edge-probability map (uint8 0..255)."""

    name = "pidinet"

    def __init__(self, weights: str = DEFAULT_WEIGHTS):
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                "PiDiNet weights not found. Fetch with:\n"
                "    python scripts/download_pidinet_weights.py\n"
                f"  expected: {weights}\n"
                "(table5_pidinet.pth, ~3 MB, committed in hellozhuo/pidinet; "
                "the script shallow-clones that repo and copies the pth.)"
            )
        import torch
        self.weights = weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net, self.transform = _load(self.device)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        import torch
        if frame.dtype != np.uint8:
            frame = _to_uint8(frame)
        h, w = frame.shape[:2]
        # PiDiNet wants 3-channel RGB (trained on color BSDS). stage1 now passes
        # color BGR; convert BGR->RGB for PIL. Gray (2-D) -> expand to 3ch RGB.
        if frame.ndim == 3:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            img = Image.fromarray(frame).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outs = self.net(x)
        edge = outs[-1]  # fused map, already sigmoid'd, at input HxW
        edge = torch.squeeze(edge).cpu().numpy()
        if edge.shape != (h, w):
            edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
        return (np.clip(edge, 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any numeric array to uint8 via min-max scaling."""
    if arr.dtype == np.uint8:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
