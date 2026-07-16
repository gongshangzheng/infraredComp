#!/usr/bin/env python3
"""infraredComp 真实训练循环 — CompressAI/ELIC image RD + ssf2020 video RD 训练。

图像路径：实例化 CompressAI image_models[name](quality, pretrained=False) 或 ELICModel
（fresh，可训练），在 FLIR thermal_16_bit / OSU 帧上做 RD 训练
（loss = λ·bpp + MSE_distortion）。

视频路径（--model ssf2020）：实例化 compressai.zoo.video_models["ssf2020"]，
默认 warm-start（pretrained=True）再在 contour 帧序列上 fine-tune（轮廓数据 OOD，
from-scratch 效果差）。forward(frames_list) 返回每帧 x_hat + likelihoods，
聚合 RD loss = λ·(全帧 bpp) + 每帧 MSE 均值。

写 results/training/{metrics.json, checkpoints/{run_id}.pth, logs/{run_id}.log}。

checkpoint→eval：存的 state_dict 可被 benchmark/learned.py:_load_model(…, checkpoint_path=…)
或 elic_model.py:load_elic_model(…, checkpoint_path=…) 或
benchmark/video/codecs/ssf2020.py(checkpoint_path=…) 加载（键名匹配，同一 model 类）。

用法（前端 POST /api/training/run 触发）:
  # image
  python3 scripts/train_model.py --model cheng2020-attn --quality 1 \
    --dataset flir/train --epochs 2 --lr 1e-4 --batch 4 --lambda 0.01 \
    --device cpu --run-id <id>
  # video (ssf2020, warm-start from pretrained, fine-tune on contour sequences)
  python3 scripts/train_model.py --model ssf2020 --quality 5 \
    --dataset datasets/contour/seq1/canny --seq-len 4 --epochs 2 \
    --lr 1e-5 --batch 1 --lambda 0.01 --device cpu --run-id <id>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# 仓库根 + benchmark 可 import
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# infraredComp 训练产物路径（与 server.config 一致）
TRAINING_DIR = REPO / "results" / "training"
CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"
LOGS_DIR = TRAINING_DIR / "logs"
METRICS_JSON = TRAINING_DIR / "metrics.json"
DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(REPO / "datasets")))


def log(run_id: str, msg: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / f"{run_id}.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


# ---- 数据集（thermal 帧 -> [0,1] 3 通道张量）---------------------------- #

IMAGE_EXTS = (".png", ".tiff", ".tif", ".jpg", ".jpeg")


class ThermalFrameDataset(Dataset):
    """从 FLIR thermal_16_bit split 或 OSU 抽帧读图，归一化 0-1 复制到 3 通道。"""

    def __init__(self, dataset_id: str, max_images: int = 64, size: int = 128):
        self.size = size
        self.files: list[Path] = []
        if dataset_id.startswith("flir/"):
            split = dataset_id.split("/", 1)[1]
            root = DATASETS_DIR / "FLIR_ADAS_1_3" / split / "thermal_16_bit"
        elif dataset_id == "osu_frames":
            # OSU 是视频；抽帧临时目录（若无则空）
            root = DATASETS_DIR / "raw" / "osu_color_thermal_frames"
        else:
            root = Path(dataset_id)
        if root.is_dir():
            self.files = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])[:max_images]
        if not self.files:
            raise RuntimeError(f"无训练图像：{root}（dataset_id={dataset_id}）")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        from PIL import Image
        import numpy as np
        img = Image.open(self.files[i])
        arr = np.array(img, dtype=np.float32)
        # 16-bit -> [0,1]
        if arr.max() > 1.5:
            arr = arr / 65535.0 if arr.dtype.kind == "u" else arr / 65535.0
        elif arr.max() > 1.5:  # 8-bit
            arr = arr / 255.0
        # 灰度 -> 3 通道（CompressAI RGB 模型）
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=0)
        else:  # HWC -> CHW
            arr = arr.transpose(2, 0, 1)
        t = torch.from_numpy(arr[:3]).float()
        if t.shape[1] != self.size or t.shape[2] != self.size:
            t = nn.functional.interpolate(t.unsqueeze(0), size=(self.size, self.size), mode="bilinear", align_corners=False).squeeze(0)
        t = torch.clamp(t, 0.0, 1.0)
        return t


def _is_video_model(model_id: str) -> bool:
    return model_id == "ssf2020"


# ---- 统一训练数据集解析（imagenet 在线 / 离线 contour 自动提取+skip）-------- #

def _ensure_contour_dir(dataset_id: str, method: str) -> Path:
    """对离线数据集（flir/osu/xiph/视频），若 contour 产物不存在则提取到
    datasets/contour/<source>/<method>/（skip-if-exists），返回该 contour 目录。

    imagenet/contour 路径直接传入的不触发提取。返回 contour 目录 Path。
    """
    from benchmark.video import config as vconfig
    from benchmark.video.stage1_extract import extract_contour_video

    # 已是 contour 目录路径或绝对目录：直接用
    if dataset_id.startswith("contour/"):
        return DATASETS_DIR / dataset_id
    cand = Path(dataset_id)
    if cand.is_absolute() and cand.is_dir():
        return cand

    # raw 数据集 → 定位 raw 输入，提取到 contour/<source>/<method>
    if dataset_id.startswith("flir/"):
        split = dataset_id.split("/", 1)[1]
        raw = DATASETS_DIR / "FLIR_ADAS_1_3" / split / "thermal_16_bit"
        source = f"flir_{split}"
    elif dataset_id == "osu_frames":
        raw = DATASETS_DIR / "raw" / "osu_color_thermal_frames"
        source = "osu_frames"
    elif dataset_id.startswith("xiph/"):
        seq = dataset_id.split("/", 1)[1]
        raw = DATASETS_DIR / "raw" / "xiph_cif" / f"{seq}.y4m"
        source = seq
    else:
        # 当作字面路径（raw 视频文件或图像目录）
        raw = Path(dataset_id)
        source = raw.stem or raw.name

    out_dir = vconfig.CONTOUR_DIR / source / method
    if not (out_dir / "manifest.json").is_file():
        log("extract", f"[extract] {dataset_id} -> contour/{source}/{method}（method={method}）")
    artifact = extract_contour_video(raw, method=method, skip_if_exists=True)
    return Path(artifact.frames_dir)


def resolve_training_dataset(dataset_id: str, method: str, max_images: int, size: int,
                             shards: int, *, is_video: bool, seq_len: int = 4,
                             max_sequences: int = 64, num_workers: int = 0,
                             out_channels: int = 3) -> Dataset:
    """按 --dataset 前缀分发到合适的训练 Dataset。

    - imagenet-<split>        → ImageNetContourDataset（parquet 流式在线提边缘，不落地）
    - bsds-<split>            → ContourPNGDataset（BSDS500 GT 软边缘 PNG，见 convert_bsds_gt.py）
    - 其余（flir/osu/xiph/contour/视频/目录）→ 先确保 contour 目录，再读 PNG 帧
    """
    if dataset_id.startswith("bsds-"):
        from scripts.imagenet_contour_dataset import ContourPNGDataset
        if is_video:
            raise RuntimeError("BSDS GT 仅支持单帧图像压缩模型")
        split = dataset_id.split("-", 1)[1]
        png_dir = DATASETS_DIR / "contour" / f"bsds_{split}_gt"
        if not (png_dir / "manifest.json").is_file():
            raise RuntimeError(
                f"BSDS GT 目录无 manifest：{png_dir}。先跑 python scripts/convert_bsds_gt.py --splits {split}"
            )
        return ContourPNGDataset(str(png_dir), size=size, max_images=max_images, out_channels=out_channels)

    if dataset_id.startswith("imagenet-"):
        from scripts.imagenet_contour_dataset import (
            ImageNetContourDataset, ContourPNGDataset, split_from_dataset_id,
            preextracted_contour_dir,
        )
        if is_video:
            raise RuntimeError("imagenet 仅支持单帧图像压缩模型（非 ssf2020 视频模型）")
        split = split_from_dataset_id(dataset_id)
        # 优先走预提取 PNG（快、GPU 喂得满）；无则回退流式在线提取（慢，仅小批量/验证用）
        png_dir = preextracted_contour_dir(split, method)
        if (png_dir / "manifest.json").is_file():
            return ContourPNGDataset(str(png_dir), size=size, max_images=max_images, out_channels=out_channels)
        # 每个 worker 都会各自建一份 dataset 实例（Windows spawn）；按 worker 数缩 row-group
        # 缓存 cap，把显式缓存总占用压在 ~16GB 内（每 group ~101MB）。
        nw = max(1, num_workers or 1)
        rg_cap = 128 if num_workers <= 0 else max(8, 160 // nw)
        return ImageNetContourDataset(
            split=split, method=method,
            max_images=max_images, size=size, shards=shards, rg_cache_cap=rg_cap,
        )

    # 离线 contour 路径
    contour_dir = _ensure_contour_dir(dataset_id, method)
    if is_video:
        return VideoFrameSequenceDataset(str(contour_dir), seq_len=seq_len,
                                         max_sequences=max_sequences, size=size)
    return ThermalFrameDataset(str(contour_dir), max_images=max_images, size=size)



class VideoFrameSequenceDataset(Dataset):
    """contour 帧序列 -> list[[3,H,W] float [0,1]]（每帧 pad 到 ÷64）。

    读 datasets/contour/<source>[/<method>]/frame_*.png，按文件名排序后每
    `seq_len` 连续帧组成一个 sample（一个序列）。shuffle 由 DataLoader 控制。
    每帧用 benchmark.learned._img_to_tensor（gray→3ch + min-max 归一化）转
    [1,3,H,W]，resize 到 `size`，再 _pad_to_multiple(64)，最后 squeeze 到 [3,H,W]
    交给训练循环（循环里 unsqueeze(0) 回 [1,3,H,W] 喂 ssf2020.forward）。
    """

    def __init__(self, dataset_id: str, seq_len: int = 4, max_sequences: int = 64, size: int = 128):
        self.seq_len = max(1, seq_len)
        self.size = size
        root = Path(dataset_id)
        if not root.is_dir():
            raise RuntimeError(f"视频数据集目录不存在：{root}（dataset_id={dataset_id}）")
        files = sorted(
            p for p in root.rglob("*")
            if p.name.lower().startswith("frame_") and p.suffix.lower() in IMAGE_EXTS
        )
        if len(files) < self.seq_len:
            raise RuntimeError(f"帧数不足：{len(files)} < seq_len={self.seq_len}（root={root}）")
        # 滑窗：每 seq_len 帧一个 sample，不重叠以避免重复
        self.sequences: list[list[Path]] = [
            files[i:i + self.seq_len] for i in range(0, len(files) - self.seq_len + 1, self.seq_len)
        ]
        self.sequences = self.sequences[:max_sequences]
        if not self.sequences:
            raise RuntimeError(f"无法组装序列：{root}（files={len(files)}, seq_len={self.seq_len}）")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, i: int) -> list[torch.Tensor]:
        from PIL import Image
        import numpy as np
        from benchmark.learned import _img_to_tensor, _pad_to_multiple
        frames: list[torch.Tensor] = []
        for p in self.sequences[i]:
            arr = np.array(Image.open(p))
            t, _ = _img_to_tensor(arr)                       # (1,3,H,W) float [0,1]
            if t.shape[2] != self.size or t.shape[3] != self.size:
                t = nn.functional.interpolate(
                    t, size=(self.size, self.size), mode="bilinear", align_corners=False,
                )
            t, _ = _pad_to_multiple(t, 64)                   # ÷64
            t = torch.clamp(t, 0.0, 1.0).squeeze(0)          # (3,H,W)
            frames.append(t)
        return frames


def _seq_collate(batch: list) -> list:
    """视频 collate：不 stack，直接返回 list[sample]，每个 sample=list[[3,H,W]]。"""
    return batch


# ---- 模型实例化（fresh 可训练，与 eval 加载同一类）--------------------- #

def build_model(model_id: str, quality: int, device: str, warm_start: bool = True):
    """实例化可训练模型。ssf2020(video) 默认 warm-start；image_models/ELIC 为 fresh。"""
    if _is_video_model(model_id):
        from compressai.zoo import video_models
        m = video_models["ssf2020"](quality=quality, metric="mse", pretrained=warm_start)
    elif model_id == "ELIC":
        from benchmark.elic_model import ELICModel  # type: ignore
        m = ELICModel(N=192, M=320, num_slices=5)
    elif model_id == "difftok":
        from omegaconf import OmegaConf
        from third_party.diffTok.src.nets.contour_vqae import ContourVQAE
        cfg_path = REPO / "configs" / "difftok" / "bsds_contour.yaml"
        cfg = OmegaConf.load(cfg_path)
        m = ContourVQAE(cfg)
    else:
        from compressai.zoo import image_models
        m = image_models[model_id](quality=quality, pretrained=False)
    return m.to(device)


# ---- RD loss ------------------------------------------------------------ #

def rd_loss(out, x, lam: float) -> tuple[torch.Tensor, float, float, float]:
    """rate-distortion: loss = λ·bpp + MSE。返回 (loss, loss_val, psnr, bpp)。"""
    x_hat = out["x_hat"]
    likelihoods = out["likelihoods"]
    ll_iter = likelihoods.values() if isinstance(likelihoods, dict) else likelihoods
    num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
    # rate = -log2(likelihood) per pixel（信息量）
    rate = 0.0
    for ll in ll_iter:
        rate = rate + (-torch.log(ll + 1e-10) / math.log(2)).sum()
    bpp = (rate / num_pixels).item()
    dist = torch.mean((x_hat - x) ** 2)
    mse = dist.item()
    loss = lam * (rate / num_pixels) + dist
    psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
    return loss, loss.item(), psnr, bpp


def rd_loss_difftok(logits, x0, vq_loss, pos_weight: float = 10.0) -> tuple:
    """BCE loss for binary contour maps + VQ commitment loss.

    pos_weight upweights edge pixels to compensate class imbalance (~5-15% edges).
    Returns (loss, loss_val, psnr_proxy, bpp_dummy) matching rd_loss tuple shape.
    """
    import torch.nn.functional as F
    pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, x0, pos_weight=pw)
    loss = bce + vq_loss
    with torch.no_grad():
        x_hat = torch.sigmoid(logits)
        mse = torch.mean((x_hat - x0) ** 2).item()
        psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
    return loss, loss.item(), psnr, 0.0


def video_rd_loss(out, frames: list, lam: float) -> tuple[torch.Tensor, float, float, float]:
    """ssf2020 聚合 RD loss。

    out = model.forward(frames) -> {"x_hat": [t0_hat,...], "likelihoods": [lik0, lik1,...]}
      lik0 = {"keyframe": tensor}; lik_i(i>0) = {"motion": tensor, "residual": tensor}
    bpp 聚合全部帧的 keyframe/inter(motion+residual) likelihoods，除以 (num_pixels * n_frames)。
    distortion = 每帧 MSE 取均值。loss = λ·bpp + distortion。
    """
    x_hats = out["x_hat"]
    likelihoods = out["likelihoods"]
    n_frames = len(frames)
    # frames[i] = [1,3,H,W]
    num_pixels = frames[0].shape[0] * frames[0].shape[2] * frames[0].shape[3]
    total_pixels = num_pixels * n_frames

    total_rate = 0.0
    for lik in likelihoods:
        # lik 是 dict（keyframe 或 inter）；值是 likelihood 张量
        ll_iter = lik.values() if isinstance(lik, dict) else lik
        for ll in ll_iter:
            total_rate = total_rate + (-torch.log(ll + 1e-10) / math.log(2)).sum()
    bpp = (total_rate / total_pixels).item()

    total_dist = 0.0
    for xh, x in zip(x_hats, frames):
        total_dist = total_dist + torch.mean((xh - x) ** 2)
    dist = total_dist / n_frames
    mse = dist.item()
    loss = lam * (total_rate / total_pixels) + dist
    psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
    return loss, loss.item(), psnr, bpp


# ---- 训练中定期 eval + 可视化（held-out val）--------------------------- #

VIZ_DIR = TRAINING_DIR / "viz"  # results/training/viz/<run_id>/epoch_XXX.png


def _eval_png_dir(dataset_id: str, split: str, method: str) -> Path:
    """数据集感知的 held-out PNG 目录：BSDS GT → bsds_<split>_gt；imagenet → imagenet_<split>_<method>。"""
    if dataset_id.startswith("bsds-"):
        return DATASETS_DIR / "contour" / f"bsds_{split}_gt"
    from scripts.imagenet_contour_dataset import preextracted_contour_dir
    return preextracted_contour_dir(split, method)


def _load_split_sample(dataset_id: str, split: str, method: str, n_take: int, size: int, what: str,
                       out_channels: int = 3):
    """从某个 split 的 PNG 目录取前 n_take 张固定样本。"""
    from scripts.imagenet_contour_dataset import ContourPNGDataset
    d = _eval_png_dir(dataset_id, split, method)
    if not (d / "manifest.json").is_file():
        hint = (f"python scripts/convert_bsds_gt.py --splits {split}" if dataset_id.startswith("bsds-")
                else f"python scripts/extract_imagenet_contour.py --split {split} --method {method}")
        raise RuntimeError(f"{what} 目录无 manifest：{d}。先跑 {hint}")
    ds = ContourPNGDataset(str(d), size=size, max_images=0, out_channels=out_channels)
    if len(ds) == 0:
        raise RuntimeError(f"{what} 数据集为空：{d}")
    return [ds[i] for i in range(min(n_take, len(ds)))]


def _load_eval_samples(dataset_id: str, method: str, eval_split: str, viz_split: str,
                       eval_samples: int, viz_samples: int, size: int, out_channels: int = 3):
    """加载 held-out eval（eval_split）+ 可视化（viz_split）两份固定样本 + 原图。
    BSDS：eval=test、viz=val；imagenet：两者通常都 val（--viz-split 默认 = --eval-split）。
    """
    eval_sample = _load_split_sample(dataset_id, eval_split, method, eval_samples, size, "eval", out_channels)
    viz_sample = _load_split_sample(dataset_id, viz_split, method, viz_samples, size, "viz", out_channels)
    originals = _load_originals(dataset_id, viz_split, len(viz_sample), size)
    return eval_sample, viz_sample, originals


def _load_originals(dataset_id: str, split: str, n: int, size: int):
    """可视化用的原始自然图（三图对照的左图）。BSDS → sorted images/<split>/*.jpg（与
    bsds_<split>_gt/frame_<i> 同序同 ID 对齐）；非 BSDS 暂无来源 → None（回退两图）。"""
    if not dataset_id.startswith("bsds-"):
        return None
    from PIL import Image
    jpgs = sorted((DATASETS_DIR / "BSDS500" / "images" / split).glob("*.jpg"))
    out = []
    for j in jpgs[:n]:
        arr = np.array(Image.open(j).convert("RGB").resize((size, size), Image.BILINEAR))
        out.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)  # (3,H,W) [0,1]
    return out or None


def _run_eval(model, eval_sample: list, device: str, lam: float, batch: int,
              model_id: str = "") -> dict:
    """model.eval() + no_grad，在固定 val 样本上算 loss + psnr。"""
    model.eval()
    tot_loss = tot_psnr = tot_bpp = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(eval_sample), batch):
            chunk = eval_sample[s:s + batch]
            x = torch.stack([t.to(device) for t in chunk], dim=0)
            if model_id == "difftok":
                logits, vq_result = model.forward(x)
                loss, lv, psnr, bpp = rd_loss_difftok(logits, x, vq_result["quantizer_loss"])
            else:
                out = model.forward(x)
                loss, lv, psnr, bpp = rd_loss(out, x, lam)
            tot_loss += lv; tot_psnr += psnr; tot_bpp += bpp; n += 1
    model.train()
    return {"loss": tot_loss / max(n, 1), "psnr": tot_psnr / max(n, 1), "bpp": tot_bpp / max(n, 1)}


def _save_viz(model, viz_sample: list, originals, device: str, run_id: str, epoch: int) -> list:
    """每个样本一张三图 RGB PNG（原图 | 输入边缘 | 重建），存 viz/<run>/epoch_XXX_sample_Y.png。
    originals 缺则回退两图（输入 | 重建）。返回 6 条相对路径。"""
    from PIL import Image
    model.eval()
    out_dir = VIZ_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with torch.no_grad():
        for k, t in enumerate(viz_sample):
            x = t.unsqueeze(0).to(device)                  # (1,3,H,W)
            out = model.forward(x)
            xhat = out["x_hat"][0].clamp(0, 1)              # (3,H,W)
            panels = []
            if originals is not None and k < len(originals):
                panels.append(originals[k].to(device).clamp(0, 1))  # 原图（彩色）
            panels.append(x[0].clamp(0, 1))                # 输入边缘
            panels.append(xhat)                            # 重建
            row = torch.cat(panels, dim=2)                 # (3, H, len*W)
            arr = (row.cpu().numpy() * 255.0).clip(0, 255).astype("uint8")  # (3,H,len*W)
            arr = np.transpose(arr, (1, 2, 0))            # (H, len*W, 3) RGB
            fname = f"epoch_{epoch:03d}_sample_{k}.png"
            Image.fromarray(arr, mode="RGB").save(out_dir / fname)
            paths.append({"epoch": epoch, "sample": k, "path": f"viz/{run_id}/{fname}"})
    model.train()
    return paths


# ---- metrics.json 读写 ------------------------------------------------- #

def load_metrics() -> dict:
    if METRICS_JSON.exists():
        try:
            return json.loads(METRICS_JSON.read_text())
        except json.JSONDecodeError:
            pass
    return {"generated_at": None, "runs": []}


def _resolve_ckpt_path(p: str) -> Path:
    """checkpoint 路径解析：允许传 run_id 或相对 checkpoints/xxx 或绝对路径。"""
    pp = Path(p)
    if pp.is_file():
        return pp
    # 形如 checkpoints/ELIC__bsds__xxx.pth 或 ELIC__bsds__xxx.pth
    cand = CHECKPOINTS_DIR / (pp.name if pp.name.endswith(".pth") else f"{pp.name}.pth")
    if cand.is_file():
        return cand
    cand2 = CHECKPOINTS_DIR / str(p).replace("\\", "/").split("checkpoints/")[-1]
    if cand2.is_file():
        return cand2
    raise FileNotFoundError(f"checkpoint 不存在：{p}（checked {pp}, {cand}）")


def load_checkpoint(model, ckpt_path: str, *, strict: bool = False) -> None:
    """加载 model.state_dict（eval 的 _load_model 同语义，strict=False 容错）。
    ELICModel.load_state_dict 不接受 strict kwarg → 回退不带 strict。"""
    p = _resolve_ckpt_path(ckpt_path)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(sd, strict=strict)
    except TypeError:
        model.load_state_dict(sd)


def resume_from_run(prev_run_id: str) -> dict:
    """从 metrics.json 找旧 run 记录，返回 {loss_series, test_metrics, viz, start_epoch, best_psnr} 供续训继承。"""
    data = load_metrics()
    for r in data.get("runs", []):
        if r.get("id") == prev_run_id:
            ls = r.get("loss_series", [])
            last_ep = max((p["epoch"] for p in ls), default=0)
            best_psnr = _load_ckpt_meta(prev_run_id).get("best", {}).get("test", {}).get("psnr", -float("inf"))
            return {
                "loss_series": list(ls),
                "test_metrics": list(r.get("test_metrics", [])),
                "viz": list(r.get("viz", [])),
                "start_epoch": last_ep + 1,
                "best_psnr": best_psnr,
            }
    raise RuntimeError(f"resume 找不到旧 run 记录：{prev_run_id}（在 metrics.json）")


def _ckpt_meta_path(run_id: str) -> Path:
    return CHECKPOINTS_DIR / f"{run_id}.ckpt.json"


def _load_ckpt_meta(run_id: str) -> dict:
    p = _ckpt_meta_path(run_id)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    """原子写 JSON（utf-8）。Windows 下 os.replace 可能因别的进程在读而 Access Denied → 重试。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for _ in range(8):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.15)
    # 仍失败 → 硬写（非原子，但好过丢数据）
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_ckpt_meta(run_id: str, latest: dict | None, best: dict | None) -> None:
    """配套 json：存 latest + best checkpoint 的 epoch/train/test 指标（覆盖写）。"""
    _atomic_write_json(_ckpt_meta_path(run_id), {"run_id": run_id, "latest": latest, "best": best})


def save_metrics(data: dict) -> None:
    # 原子写（utf-8 + os.replace 重试）：训练子进程每 epoch 写、server 读并发
    _atomic_write_json(METRICS_JSON, data)


# ---- 主循环 ------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="CompressAI name 或 ELIC")
    ap.add_argument("--quality", type=int, default=3)
    ap.add_argument("--dataset", default="flir/train")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lambda", dest="lamb", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-id", dest="run_id", required=True)
    ap.add_argument("--max-images", type=int, default=0, help="<=0 = 不采样、全量参与（默认）；>0 则等间隔采样到该数")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--method", default="canny", help="轮廓提取方法 canny/sobel/hed（离线提取与 imagenet 在线共用）")
    ap.add_argument("--shards", type=int, default=0, help="imagenet parquet shard 数；<=0 = 全部 shard（默认，即全量）")
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=0, help="DataLoader 工作进程数（imagenet 全量推荐 4-8，0=主线程）")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"], help="优化器（默认 AdamW）")
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=50, help="每 N epoch 存一次 checkpoint（便于挂了 resume；0=只在结束存）")
    # 从 checkpoint 续训（二选一，互斥；都建新 run_id）
    ap.add_argument("--load", default=None, help="checkpoint 路径/run_id：加载权重 warm-start，fresh optimizer，epoch 1..N")
    ap.add_argument("--resume", default=None, help="checkpoint 路径/run_id：加载权重+继承旧 run 曲线，epoch 从 last+1 续跑 N 个")
    # 训练中定期 eval + 可视化（held-out val）
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=1, help="每 N epoch 跑一次 test eval+可视化（0=关）")
    ap.add_argument("--eval-split", dest="eval_split", default="val", help="eval 用的 held-out split（imagenet:val；BSDS:test）")
    ap.add_argument("--viz-split", dest="viz_split", default=None, help="可视化用的 split（默认 = --eval-split；BSDS 用 val）")
    ap.add_argument("--eval-samples", dest="eval_samples", type=int, default=512, help="每次 eval 的固定 val 图数")
    ap.add_argument("--viz-samples", dest="viz_samples", type=int, default=6, help="可视化固定图数（每 epoch 同一批，看重建演变）")
    # video (ssf2020) 专用
    ap.add_argument("--seq-len", type=int, default=4, help="视频序列长度（ssf2020）")
    ap.add_argument("--max-sequences", type=int, default=64, help="最大序列数（ssf2020）")
    ap.add_argument(
        "--warm-start", dest="warm_start", action="store_true", default=True,
        help="ssf2020 用 pretrained 初始化再 fine-tune（默认开）",
    )
    ap.add_argument(
        "--no-warm-start", dest="warm_start", action="store_false",
        help="关闭 warm-start（ssf2020 从 scratch 训练）",
    )
    args = ap.parse_args()

    is_video = _is_video_model(args.model)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    run_id = args.run_id
    started = time.time()

    log(run_id, f"[train] start run_id={run_id} model={args.model} q={args.quality} dataset={args.dataset} "
                f"epochs={args.epochs} device={device} warm_start={args.warm_start if is_video else 'n/a'}"
                + (f" seq_len={args.seq_len}" if is_video else ""))

    # resume 曲线继承（必须在下面 running 记录之前，否则先 wipe 掉旧曲线 → 继承 0）
    start_epoch = 1
    resumed_ls, resumed_tm, resumed_viz = [], [], []
    best_psnr = -float("inf")  # best checkpoint 判据：最高 test PSNR；resume 时从 ckpt.json 继承
    if args.resume:
        prev_id = Path(args.resume).stem if str(args.resume).endswith(".pth") else args.resume
        info = resume_from_run(prev_id)
        resumed_ls, resumed_tm, resumed_viz = info["loss_series"], info["test_metrics"], info["viz"]
        start_epoch = info["start_epoch"]
        best_psnr = info.get("best_psnr", -float("inf"))
        log(run_id, f"[train] resumed curves from {prev_id}: continue epoch {start_epoch}..{start_epoch + args.epochs - 1}, "
                    f"inherited {len(resumed_ls)} loss / {len(resumed_tm)} test / {len(resumed_viz)} viz, best_psnr={best_psnr:.2f}")

    # 训练中可见：立即记一条 running（resume 时带继承的曲线，不 wipe）
    _record_run(args, run_id, started, status="running", loss_series=resumed_ls,
                test_metrics=resumed_tm, viz=resumed_viz)

    # 1. 数据
    try:
        ds = resolve_training_dataset(
            args.dataset, args.method, args.max_images, args.size, args.shards,
            is_video=is_video, seq_len=args.seq_len, max_sequences=args.max_sequences,
            num_workers=args.num_workers,
            out_channels=1 if args.model == "difftok" else 3,
        )
    except (RuntimeError, KeyError, FileNotFoundError) as e:
        log(run_id, f"[train] dataset error: {e}")
        _record_run(args, run_id, started, status="failed", loss_series=[], error=str(e))
        return 1
    nw = args.num_workers
    kw = {} if nw <= 0 else {"num_workers": nw, "persistent_workers": True, "prefetch_factor": 4}
    if is_video:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=_seq_collate)
        log(run_id, f"[train] dataset: {len(ds)} sequences x {args.seq_len} frames, batch={args.batch}")
    else:
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, **kw)
        log(run_id, f"[train] dataset: {len(ds)} images, batch={args.batch}, num_workers={nw}")

    # 2. 模型
    try:
        model = build_model(args.model, args.quality, device, warm_start=args.warm_start)
    except Exception as e:
        log(run_id, f"[train] model build error: {e}")
        _record_run(args, run_id, started, status="failed", loss_series=[], error=str(e))
        return 1
    model.train()
    OptCls = torch.optim.AdamW if (args.optimizer or "adamw").lower() == "adamw" else torch.optim.Adam
    optimizer = OptCls(model.parameters(), lr=args.lr)
    log(run_id, f"[train] model built, params={sum(p.numel() for p in model.parameters())}, optimizer={args.optimizer}")

    # 2a. --load 权重（resume 的曲线继承已在 main 开头完成；模型从 checkpoint 加载）
    if args.load:
        load_checkpoint(model, args.load)
        log(run_id, f"[train] loaded weights from {args.load}")

    # 2b. held-out eval + 可视化样本（仅图像模型；每 epoch 末跑）
    eval_sample, viz_sample, originals, test_metrics, viz = None, None, None, [], []
    if args.resume:
        test_metrics, viz = list(resumed_tm), list(resumed_viz)  # 继承旧 run 曲线（loss_series 见下）
    eval_on = (not is_video) and args.eval_every and args.eval_every > 0
    if eval_on:
        try:
            viz_split = args.viz_split or args.eval_split
            eval_sample, viz_sample, originals = _load_eval_samples(
                args.dataset, args.method, args.eval_split, viz_split,
                args.eval_samples, args.viz_samples, args.size,
                out_channels=1 if args.model == "difftok" else 3)
            log(run_id, f"[train] eval on {args.dataset}/{args.eval_split} ({len(eval_sample)} imgs), "
                        f"viz on {args.dataset}/{viz_split} ({len(viz_sample)} imgs), every {args.eval_every} epoch"
                        + ("" if originals is None else f", with originals (3-panel)"))
        except Exception as e:
            log(run_id, f"[train] eval disabled: {e}")
            eval_on = False

    # 3. 训练循环（resume 时继承旧 loss_series + 从 start_epoch 续跑 args.epochs 个）
    loss_series: list[dict] = list(resumed_ls) if args.resume else []
    best_meta = _load_ckpt_meta(run_id).get("best") if args.resume else None  # resume 继承 best 记录
    latest_meta = None
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            ep_loss, ep_psnr, ep_bpp, n = 0.0, 0.0, 0.0, 0
            if is_video:
                for batch in loader:  # batch = list[sample], sample = list[[3,H,W]]
                    for frames_3hw in batch:
                        frames = [t.unsqueeze(0).to(device) for t in frames_3hw]  # [1,3,H,W]
                        optimizer.zero_grad()
                        out = model.forward(frames)
                        loss, lv, psnr, bpp = video_rd_loss(out, frames, args.lamb)
                        loss.backward()
                        optimizer.step()
                        ep_loss += lv; ep_psnr += psnr; ep_bpp += bpp; n += 1
            else:
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    if args.model == "difftok":
                        logits, vq_result = model(batch)
                        loss, lv, psnr, bpp = rd_loss_difftok(logits, batch, vq_result["quantizer_loss"])
                    else:
                        out = model(batch)
                        loss, lv, psnr, bpp = rd_loss(out, batch, args.lamb)
                    loss.backward()
                    optimizer.step()
                    ep_loss += lv; ep_psnr += psnr; ep_bpp += bpp; n += 1
            avg = {"epoch": epoch, "loss": ep_loss / max(n, 1), "psnr": ep_psnr / max(n, 1), "bpp": ep_bpp / max(n, 1)}
            loss_series.append(avg)
            ep_i = epoch - start_epoch + 1
            log(run_id, f"[train] epoch {epoch} ({ep_i}/{args.epochs}) loss={avg['loss']:.4f} psnr={avg['psnr']:.2f} bpp={avg['bpp']:.3f}")
            # 训练中 eval + 可视化（每 eval_every epoch）
            if eval_on and (epoch % args.eval_every == 0):
                try:
                    em = _run_eval(model, eval_sample, device, args.lamb, args.batch, model_id=args.model)
                    em["epoch"] = epoch
                    test_metrics.append(em)
                    vpaths = _save_viz(model, viz_sample, originals, device, run_id, epoch)
                    viz.extend(vpaths)  # 每 epoch 6 条 {epoch, sample, path}
                    log(run_id, f"[train]   eval epoch {epoch}: test psnr={em['psnr']:.2f} bpp={em['bpp']:.3f} "
                                f"loss={em['loss']:.4f} | viz {len(vpaths)} panels -> viz/{run_id}/")
                    # best checkpoint：test PSNR 创新高时存 <run>.best.pth（覆盖）
                    if em["psnr"] > best_psnr:
                        best_psnr = em["psnr"]
                        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{run_id}.best.pth")
                        best_meta = {"epoch": epoch, "criterion": "test_psnr", "path": f"checkpoints/{run_id}.best.pth",
                                     "train": avg, "test": {"psnr": em["psnr"], "bpp": em["bpp"], "loss": em["loss"]},
                                     "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                        log(run_id, f"[train]   new best test psnr={em['psnr']:.2f} @ epoch {epoch} -> {run_id}.best.pth")
                except Exception as ee:
                    log(run_id, f"[train]   eval/viz error epoch {epoch}: {ee}")
            # 实时曲线：每 epoch 把当前 loss_series 落盘，前端轮询即可看到曲线增长
            _record_run(args, run_id, started, status="running", loss_series=loss_series,
                        test_metrics=test_metrics, viz=viz)
            # 定期存 latest checkpoint（覆盖）+ 配套 json（latest/best 指标）
            if args.ckpt_every and (epoch % args.ckpt_every == 0):
                CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), CHECKPOINTS_DIR / f"{run_id}.pth")
                last_tm = test_metrics[-1] if test_metrics else None
                latest_meta = {"epoch": epoch, "path": f"checkpoints/{run_id}.pth", "train": avg,
                               "test": last_tm, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                _write_ckpt_meta(run_id, latest_meta, best_meta)
    except Exception as e:
        log(run_id, f"[train] training loop error: {e}")
        _record_run(args, run_id, started, status="failed", loss_series=loss_series,
                    test_metrics=test_metrics, viz=viz, error=str(e))
        return 1

    # 4. 存 checkpoint（state_dict，可被 _load_model/checkpoint_path 加载）
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / f"{run_id}.pth"
    torch.save(model.state_dict(), ckpt_path)
    log(run_id, f"[train] checkpoint saved: {ckpt_path}")

    final_loss = loss_series[-1]["loss"] if loss_series else None
    best_metric = max((p["psnr"] for p in loss_series), default=None)
    _record_run(args, run_id, started, status="completed",
                loss_series=loss_series, checkpoint_path=f"checkpoints/{run_id}.pth",
                final_loss=final_loss, best_metric=best_metric,
                test_metrics=test_metrics, viz=viz)
    log(run_id, f"[train] done run_id={run_id}")
    return 0


def _record_run(args, run_id: str, started: float, *, status: str, loss_series, **kw) -> None:
    data = load_metrics()
    run = {
        "id": run_id,
        "model": args.model,
        "dataset": args.dataset,
        "quality": args.quality,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "lambda": args.lamb,
        "device": args.device,
        "seq_len": getattr(args, "seq_len", None),
        "warm_start": getattr(args, "warm_start", None),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S") if status != "running" else None,
        "status": status,
        "loss_series": loss_series,
        "test_metrics": kw.get("test_metrics", []),
        "viz": kw.get("viz", []),
        "final_loss": kw.get("final_loss"),
        "best_metric": kw.get("best_metric"),
        "checkpoint_path": kw.get("checkpoint_path"),
        "error": kw.get("error"),
    }
    # 替换同 id run（重跑）或追加
    runs = [r for r in data.get("runs", []) if r.get("id") != run_id]
    runs.append(run)
    data["runs"] = runs
    data["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_metrics(data)


if __name__ == "__main__":
    raise SystemExit(main())
