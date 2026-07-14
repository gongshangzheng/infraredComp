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
                             max_sequences: int = 64, num_workers: int = 0) -> Dataset:
    """按 --dataset 前缀分发到合适的训练 Dataset。

    - imagenet-<split>        → ImageNetContourDataset（parquet 流式在线提边缘，不落地）
    - 其余（flir/osu/xiph/contour/视频/目录）→ 先确保 contour 目录，再读 PNG 帧
    """
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
            return ContourPNGDataset(str(png_dir), size=size, max_images=max_images)
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
    # 原子写：训练子进程每 epoch 写、server 读并发，避免读到半截 JSON
    tmp = METRICS_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, METRICS_JSON)


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

    # 训练中可见：立即记一条 running（前端曲线页能看到该 run 正在跑）
    _record_run(args, run_id, started, status="running", loss_series=[])

    # 1. 数据
    try:
        ds = resolve_training_dataset(
            args.dataset, args.method, args.max_images, args.size, args.shards,
            is_video=is_video, seq_len=args.seq_len, max_sequences=args.max_sequences,
            num_workers=args.num_workers,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    log(run_id, f"[train] model built, params={sum(p.numel() for p in model.parameters())}")

    # 3. 训练循环
    loss_series: list[dict] = []
    try:
        for epoch in range(1, args.epochs + 1):
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
                    out = model(batch)
                    loss, lv, psnr, bpp = rd_loss(out, batch, args.lamb)
                    loss.backward()
                    optimizer.step()
                    ep_loss += lv; ep_psnr += psnr; ep_bpp += bpp; n += 1
            avg = {"epoch": epoch, "loss": ep_loss / max(n, 1), "psnr": ep_psnr / max(n, 1), "bpp": ep_bpp / max(n, 1)}
            loss_series.append(avg)
            log(run_id, f"[train] epoch {epoch}/{args.epochs} loss={avg['loss']:.4f} psnr={avg['psnr']:.2f} bpp={avg['bpp']:.3f}")
            # 实时曲线：每 epoch 把当前 loss_series 落盘，前端轮询即可看到曲线增长
            _record_run(args, run_id, started, status="running", loss_series=loss_series)
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
        "seq_len": getattr(args, "seq_len", None),
        "warm_start": getattr(args, "warm_start", None),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S") if status != "running" else None,
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
