#!/usr/bin/env python3
"""infraredComp 真实训练循环 — CompressAI/ELIC rate-distortion 训练。

实例化 CompressAI image_models[name](quality, pretrained=False) 或 ELICModel（fresh，可训练），
在 FLIR thermal_16_bit / OSU 帧上做 RD 训练（loss = λ·bpp + MSE_distortion），
写 results/training/{metrics.json, checkpoints/{run_id}.pth, logs/{run_id}.log}。

checkpoint→eval：存的 state_dict 可被 benchmark/learned.py:_load_model(…, checkpoint_path=…)
或 elic_model.py:load_elic_model(…, checkpoint_path=…) 加载（键名匹配，同一 model 类）。

用法（前端 POST /api/training/run 触发）:
  python3 scripts/train_model.py --model cheng2020-attn --quality 1 \
    --dataset flir/train --epochs 2 --lr 1e-4 --batch 4 --lambda 0.01 \
    --device cpu --run-id <id>
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


# ---- 模型实例化（fresh 可训练，与 eval 加载同一类）--------------------- #

def build_model(model_id: str, quality: int, device: str):
    """实例化 fresh 可训练模型。CompressAI zoo 或 ELIC。"""
    if model_id == "ELIC":
        from benchmark.elic_model import ELICModel  # type: ignore
        m = ELICModel(N=192, M=320, num_slices=5)
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


# ---- metrics.json 读写 ------------------------------------------------- #

def load_metrics() -> dict:
    if METRICS_JSON.exists():
        try:
            return json.loads(METRICS_JSON.read_text())
        except json.JSONDecodeError:
            pass
    return {"generated_at": None, "runs": []}


def save_metrics(data: dict) -> None:
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))


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
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    run_id = args.run_id
    started = time.time()

    log(run_id, f"[train] start run_id={run_id} model={args.model} q={args.quality} dataset={args.dataset} epochs={args.epochs} device={device}")

    # 1. 数据
    try:
        ds = ThermalFrameDataset(args.dataset, max_images=args.max_images, size=args.size)
    except RuntimeError as e:
        log(run_id, f"[train] dataset error: {e}")
        _record_run(args, run_id, started, status="failed", loss_series=[], error=str(e))
        return 1
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    log(run_id, f"[train] dataset: {len(ds)} images, batch={args.batch}")

    # 2. 模型
    try:
        model = build_model(args.model, args.quality, device)
    except Exception as e:
        log(run_id, f"[train] model build error: {e}")
        _record_run(args, run_id, started, status="failed", loss_series=[], error=str(e))
        return 1
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log(run_id, f"[train] model built, params={sum(p.numel() for p in model.parameters())}")

    # 3. 训练循环
    loss_series: list[dict] = []
    try:
        for epoch in range(1, args.epochs + 1):
            ep_loss, ep_psnr, ep_bpp, n = 0.0, 0.0, 0.0, 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss, lv, psnr, bpp = rd_loss(out, batch, args.lamb)
                loss.backward()
                optimizer.step()
                ep_loss += lv; ep_psnr += psnr; ep_bpp += bpp; n += 1
            avg = {"epoch": epoch, "loss": ep_loss / max(n, 1), "psnr": ep_psnr / max(n, 1), "bpp": ep_bpp / max(n, 1)}
            loss_series.append(avg)
            log(run_id, f"[train] epoch {epoch}/{args.epochs} loss={avg['loss']:.4f} psnr={avg['psnr']:.2f} bpp={avg['bpp']:.3f}")
    except Exception as e:
        log(run_id, f"[train] training loop error: {e}")
        _record_run(args, run_id, started, status="failed", loss_series=loss_series, error=str(e))
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
                final_loss=final_loss, best_metric=best_metric)
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
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "loss_series": loss_series,
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
