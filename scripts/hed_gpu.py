#!/usr/bin/env python3
"""GPU 版 HED（PyTorch）—— 复刻 s9xie/hed 架构，权重从 caffemodel 转移，跑在 CUDA。

本机 cv2.dnn 无 CUDA，CPU 跑 1.28M 张 imagenet ≈ 20h；此模块用 torch 在 5090 上
批量推理（~30min）。权重沿用 third_party/hed 的 deploy.prototxt + caffemodel
（经 cv2.dnn 读 blobs 转成 torch，无需 caffe）。

输出与 cv2 版 hed 对齐（同模型同权重，仅推理后端不同），可直接作为 'hed' 边缘。
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

_HED_DIR = REPO / "third_party" / "hed"
PROTOTXT = str(_HED_DIR / "deploy.prototxt")
CAFFEMODEL = str(_HED_DIR / "hed_pretrained_bsds.caffemodel")
# ImageNet per-channel means (BGR) —— s9xie/hed recipe
_MEAN_BGR = (104.00698793, 116.66876762, 122.67891434)

# cv2 layer 名 -> torch 模块名（score-dsn1 的 conv 在 cv2 里叫 'score-dsn1-up'；
# upsample_* 是各 side-output 的固定双线性 deconv 核，也转移以精确对齐 cv2）
_CV2_TO_TORCH = {
    **{f"conv{n}": f"conv{n}" for n in
       ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2", "3_3", "4_1", "4_2", "4_3", "5_1", "5_2", "5_3"]},
    "score-dsn1-up": "dsn1", "score-dsn2": "dsn2", "score-dsn3": "dsn3",
    "score-dsn4": "dsn4", "score-dsn5": "dsn5", "upscore-fuse": "fuse",
    "upsample_2": "up2", "upsample_4": "up4", "upsample_8": "up8", "upsample_16": "up16",
}


class HEDTorch(nn.Module):
    """s9xie/hed：VGG trunk（conv1_1 pad=35）+ 5 个 1×1 side-output + 1×1 融合。
    上采样用 bilinear interpolate（等价 caffemodel 里固定的双线性 deconv 核）。"""

    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.dsn1 = nn.Conv2d(64, 1, 1)
        self.dsn2 = nn.Conv2d(128, 1, 1)
        self.dsn3 = nn.Conv2d(256, 1, 1)
        self.dsn4 = nn.Conv2d(512, 1, 1)
        self.dsn5 = nn.Conv2d(512, 1, 1)
        self.fuse = nn.Conv2d(5, 1, 1)
        # side-output 上采样：用 caffemodel 里固定的双线性 deconv 核（精确对齐 cv2）
        self.up2 = nn.ConvTranspose2d(1, 1, 4, stride=2)
        self.up4 = nn.ConvTranspose2d(1, 1, 8, stride=4)
        self.up8 = nn.ConvTranspose2d(1, 1, 16, stride=8)
        self.up16 = nn.ConvTranspose2d(1, 1, 32, stride=16)
        self.pool = nn.MaxPool2d(2, 2)

    @staticmethod
    def _center_crop(t: torch.Tensor, H: int, W: int) -> torch.Tensor:
        _, _, h, w = t.shape
        y1 = (h - H) // 2
        x1 = (w - W) // 2
        return t[:, :, y1:y1 + H, x1:x1 + W]

    def _side(self, d: torch.Tensor, up: nn.Module, H: int, W: int) -> torch.Tensor:
        return self._center_crop(up(d), H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        h = F.relu(self.conv1_1(x)); h = F.relu(self.conv1_2(h)); d1 = self.dsn1(h); p = self.pool(h)
        h = F.relu(self.conv2_1(p)); h = F.relu(self.conv2_2(h)); d2 = self.dsn2(h); p = self.pool(h)
        h = F.relu(self.conv3_1(p)); h = F.relu(self.conv3_2(h)); h = F.relu(self.conv3_3(h)); d3 = self.dsn3(h); p = self.pool(h)
        h = F.relu(self.conv4_1(p)); h = F.relu(self.conv4_2(h)); h = F.relu(self.conv4_3(h)); d4 = self.dsn4(h); p = self.pool(h)
        h = F.relu(self.conv5_1(p)); h = F.relu(self.conv5_2(h)); h = F.relu(self.conv5_3(h)); d5 = self.dsn5(h)
        # dsn1 无 deconv（conv1_2 已是 ~输入分辨率，直接 center-crop）
        s1 = self._center_crop(d1, H, W)
        s2 = self._side(d2, self.up2, H, W); s3 = self._side(d3, self.up4, H, W)
        s4 = self._side(d4, self.up8, H, W); s5 = self._side(d5, self.up16, H, W)
        return torch.sigmoid(self.fuse(torch.cat([s1, s2, s3, s4, s5], dim=1)))  # (N,1,H,W) in [0,1]


def load_hed_from_caffe(model: HEDTorch, prototxt: str = PROTOTXT, caffemodel: str = CAFFEMODEL) -> None:
    """用 cv2.dnn 读 caffemodel 的 conv 权重，拷进 torch 模型（无需 caffe）。"""
    import cv2
    from benchmark.video.extractors.hed import _ensure_crop_layer_registered
    _ensure_crop_layer_registered()
    net = cv2.dnn.readNet(prototxt, caffemodel)
    sd = {}
    with torch.no_grad():
        for cvname, tname in _CV2_TO_TORCH.items():
            lid = net.getLayerId(cvname)
            L = net.getLayer(lid)
            sd[f"{tname}.weight"] = torch.from_numpy(L.blobs[0].copy())
            sd[f"{tname}.bias"] = torch.from_numpy(L.blobs[1].reshape(-1).copy())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"HED weight transfer mismatch: missing={missing} unexpected={unexpected}")


def _gray_to_bgr_batch(imgs: list[np.ndarray], size: int, mean_bgr, device: str) -> torch.Tensor:
    """list[uint8 HxW 或 HxWx3] -> (N,3,size,size) BGR - mean（不归一化到 0-1）。"""
    arrs = []
    for im in imgs:
        if im.ndim == 2:
            bgr = np.stack([im, im, im], axis=-1)  # gray -> BGR（HWC, B=G=R）
        else:
            bgr = im[..., :3]
            if bgr.shape[-1] == 3 and bgr is im:  # 防御性 copy
                bgr = bgr.copy()
        pil = Image.fromarray(bgr).resize((size, size), Image.BILINEAR)
        arrs.append(np.asarray(pil))
    batch = np.stack(arrs).astype(np.float32)            # (N,H,W,3)
    batch = batch[..., ::-1] if False else batch          # 已是 BGR
    batch = np.transpose(batch, (0, 3, 1, 2))             # (N,3,H,W)
    t = torch.from_numpy(batch)
    t = t - torch.tensor(mean_bgr, dtype=torch.float32).view(1, 3, 1, 1)
    return t.to(device)


def hed_edges_batch(model: HEDTorch, imgs: list[np.ndarray], size: int = 256,
                    device: str = "cuda", mean_bgr=_MEAN_BGR) -> np.ndarray:
    """批量推理 -> (N, size, size) uint8 边缘（与 cv2 hed 同语义）。"""
    if not imgs:
        return np.zeros((0, size, size), dtype=np.uint8)
    x = _gray_to_bgr_batch(imgs, size, mean_bgr, device)
    with torch.no_grad():
        out = model.forward(x)                            # (N,1,size,size) [0,1]
    return (out.squeeze(1).clamp_(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
