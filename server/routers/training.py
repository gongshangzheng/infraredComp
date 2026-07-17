"""训练体系路由 — infraredComp 定制版（接 CompressAI/ELIC + FLIR/OSU 数据）。

复用上游 ProjFlow training.py 契约（models/datasets/configs/run/runs/checkpoints/outputs），
把领域数据源接到 infraredComp：
  models    = CompressAI 6 架构 + ELIC（benchmark/learned.py + elic_model.py）
  datasets  = FLIR thermal train/val/video split + OSU 帧序列
  configs   = 默认超参 preset（epochs/lr/batch/λ/quality）
  run       = subprocess 触发 scripts/train_model.py（异步起，返回 started + run_id）
  runs      = 读 results/training/metrics.json（含 loss_series）
  checkpoints = 扫 results/training/checkpoints/*.pth
  outputs   = safe_resolve 服务 .pth / .log 文件

checkpoint→eval 打通：trained .pth 可在 EvalRun 选 DL 模型时引用（见 evaluation.py
models 契约的 checkpoint 字段 + learned.py/elic_model.py 的 checkpoint_path override）。
"""
import os
import json
import sys
import time
import subprocess

from fastapi import APIRouter, Body
from fastapi.responses import FileResponse

from server.config import (
    DATASETS_DIR, TRAINING_METRICS_JSON, CHECKPOINTS_DIR, TRAINING_OUTPUTS_DIR,
)
from server.utils.file_utils import read_file, safe_resolve
from server.cache import file_cached

router = APIRouter(prefix="/api/training", tags=["training"])

# CompressAI 架构 + ELIC（来自 benchmark/learned.py + elic_model.py）
_COMPRESSAI_MODELS = [
    {"id": "bmshj2018-factorized", "name": "Factorized Prior", "架构": "CompressAI", "qualities": [1, 4, 8]},
    {"id": "bmshj2018-hyperprior", "name": "Hyperprior", "架构": "CompressAI", "qualities": [1, 4, 8]},
    {"id": "mbt2018-mean", "name": "Mean Scale Hyperprior", "架构": "CompressAI", "qualities": [1, 4, 8]},
    {"id": "mbt2018", "name": "Scale Hyperprior", "架构": "CompressAI", "qualities": [1, 4, 8]},
    {"id": "cheng2020-anchor", "name": "Channel Autoregressive", "架构": "CompressAI", "qualities": [1, 4, 6]},
    {"id": "cheng2020-attn", "name": "Attention-guided", "架构": "CompressAI", "qualities": [1, 4, 6]},
    {"id": "ELIC", "name": "ELIC", "架构": "ELIC (CVPR2022)", "qualities": [1, 4, 5]},
    {"id": "ssf2020", "name": "SSF2020 (video)", "架构": "CompressAI video", "qualities": [1, 3, 5, 7, 9]},
]

# 训练数据集（imagenet train 在线提边缘；FLIR thermal 离线提取轮廓）
# imagenet 三 split 分工：train→训练（此处），val→评测 speed run，test→评测 formal。
_DEFAULT_DATASETS = [
    {"id": "imagenet-train", "name": "ImageNet train（在线提边缘）", "split": "train", "num_samples": "1.28M", "modalities": ["rgb"], "description": "ImageNet-1k train parquet，训练时在线提边缘轮廓（不落地），按 --shards/--max-images 采样一部分图"},
    {"id": "bsds-train", "name": "BSDS500 train (GT 软边缘)", "split": "train", "num_samples": "200", "modalities": ["edge_gt"], "description": "BSDS500 train split GT（.mat Boundaries 多标注者平均→软边缘 PNG）；train→训练 / test→eval / val→可视化（先跑 convert_bsds_gt.py）"},
    {"id": "bsds-val", "name": "BSDS500 val (GT)", "split": "val", "num_samples": "100", "modalities": ["edge_gt"], "description": "BSDS500 val split GT——用作训练可视化集"},
    {"id": "bsds-test", "name": "BSDS500 test (GT)", "split": "test", "num_samples": "200", "modalities": ["edge_gt"], "description": "BSDS500 test split GT——用作训练 eval/test 集"},
    {"id": "flir/train", "name": "FLIR thermal train", "split": "train", "num_samples": "?", "modalities": ["thermal_16bit"], "description": "FLIR ADAS thermal_16_bit 训练 split（先离线提取轮廓再训）"},
    {"id": "flir/val", "name": "FLIR thermal val", "split": "val", "num_samples": "?", "modalities": ["thermal_16bit"], "description": "FLIR ADAS thermal_16_bit 验证 split（先离线提取轮廓再训）"},
]

_DEFAULT_CONFIGS = [
    {
        "id": "default", "name": "默认 RD 训练",
        "epochs": 100, "lr": 1e-4, "batch_size": 16, "optimizer": "adamw",
        "scheduler": "cosine", "lambda": 0.01, "quality": 3,
        "description": "rate-distortion 训练（MSE + λ·bpp）；下游按模型/数据集调",
    },
    {
        "id": "fast", "name": "快速验证", "epochs": 2, "lr": 1e-4, "batch_size": 4,
        "optimizer": "adamw", "scheduler": "none", "lambda": 0.01, "quality": 1,
        "description": "2 epoch 小数据快速跑通流水线",
    },
]


def _load_metrics() -> dict:
    content = file_cached(TRAINING_METRICS_JSON, ttl=5.0)
    if not content:
        return {"generated_at": None, "runs": []}
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "runs" in data:
            return data
        return {"generated_at": None, "runs": data if isinstance(data, list) else []}
    except json.JSONDecodeError:
        return {"generated_at": None, "runs": []}


def _trained_checkpoints_for(model_id: str) -> list[str]:
    """扫 CHECKPOINTS_DIR 找该 model 的 trained checkpoint 文件名。"""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []
    out = []
    for fn in os.listdir(CHECKPOINTS_DIR):
        if fn.startswith('.') or not fn.endswith('.pth'):
            continue
        # 命名约定: {model_id}__q{quality}__{run_id}.pth
        if fn.startswith(model_id):
            out.append(f"checkpoints/{fn}")
    return sorted(out)


# ---- models ------------------------------------------------------------- #

@router.get("/models")
async def get_models():
    """可训练模型清单：CompressAI 6 架构 + ELIC，每个带 quality 级 + 已有 trained checkpoint。"""
    out = []
    for m in _COMPRESSAI_MODELS:
        out.append({
            "id": m["id"], "name": m["name"], "架构": m["架构"],
            "qualities": m["qualities"],
            "pretrained": "CompressAI zoo (~/.cache/torch/hub)" if "CompressAI" in m["架构"] else "ELIC Google Drive (~/.cache/.../elic)",
            "trained_checkpoint": _trained_checkpoints_for(m["id"]),
        })
    return out


@router.get("/models/{model_id}")
async def get_model_detail(model_id: str):
    for m in await get_models():
        if m["id"] == model_id:
            return m
    return {"detail": "Model not found"}, 404


# ---- datasets ---------------------------------------------------------- #

@router.get("/datasets")
async def get_datasets():
    """训练数据集：FLIR thermal splits + OSU 帧。补充实际样本数（若目录存在）。"""
    out = []
    for d in _DEFAULT_DATASETS:
        row = dict(d)
        # 尝试算样本数
        if d["id"].startswith("flir/"):
            split = d["split"]
            flir = os.path.join(DATASETS_DIR, "FLIR_ADAS_1_3", split, "thermal_16_bit")
            if os.path.isdir(flir):
                row["num_samples"] = len([f for f in os.listdir(flir) if f.endswith(('.png', '.tiff', '.jpg'))])
        elif d["id"] == "osu_frames":
            osu = os.path.join(DATASETS_DIR, "raw", "osu_color_thermal")
            if os.path.isdir(osu):
                row["num_samples"] = sum(1 for r in os.listdir(osu) if r.endswith('.mp4'))
        out.append(row)
    return out


@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(dataset_id: str):
    for d in await get_datasets():
        if d["id"] == dataset_id:
            return d
    return {"detail": "Dataset not found"}, 404


# ---- configs ----------------------------------------------------------- #

@router.get("/configs")
async def get_configs():
    return _DEFAULT_CONFIGS


@router.get("/configs/{config_id}")
async def get_config_detail(config_id: str):
    for c in _DEFAULT_CONFIGS:
        if c["id"] == config_id:
            return c
    return {"detail": "Config not found"}, 404


# ---- run（触发训练）---------------------------------------------------- #

@router.post("/run")
async def run_training(data: dict = Body(...)):
    """异步触发 scripts/train_model.py，立即返回 started + run_id。

    训练脚本写 results/training/metrics.json（loss_series）+ checkpoints/{run_id}.pth + logs/{run_id}.log。
    """
    # run_id 带数据集名 + 轮廓方法名（如 ELIC__bsds-train__gt__q1__<ts>、ELIC__imagenet-train__canny__q1__<ts>）
    ds_tag = str(data.get("dataset_id", "dataset")).replace("/", "_")
    # BSDS 用 GT（不是 canny/hed 提取），方法标签强制 gt；其余用所选方法
    method_tag = "gt" if ds_tag.startswith("bsds") else str(data.get("method", "canny"))
    run_id = f"{data.get('model_id', 'model')}__{ds_tag}__{method_tag}__q{data.get('quality', 0)}__{int(time.time())}"
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "train_model.py")
    args = [sys.executable, script,
            "--model", str(data.get("model_id", "")),
            "--quality", str(data.get("quality", 3)),
            "--dataset", str(data.get("dataset_id", "flir/train")),
            "--epochs", str(data.get("epochs", 100)),
            "--lr", str(data.get("lr", 1e-4)),
            "--batch", str(data.get("batch_size", 16)),
            "--lambda", str(data.get("lamb", 0.01)),
            "--device", str(data.get("device", "cuda")),
            "--optimizer", str(data.get("optimizer", "adamw")),
            "--method", method_tag,
            "--shards", str(data.get("shards", 4)),
            "--max-images", str(data.get("max_images", 0)),
            "--size", str(data.get("size", 128)),
            "--run-id", run_id]
    # 从 checkpoint 续训（load=warm-start / resume=续跑；二选一，来自前端 ckpt_mode radio）
    ckpt = data.get("checkpoint")
    if ckpt and data.get("ckpt_mode") == "load":
        args += ["--load", str(ckpt)]
    elif ckpt and data.get("ckpt_mode") == "resume":
        args += ["--resume", str(ckpt)]
    # 视频模型（ssf2020）专用参数：序列长度 / 最大序列数 / warm-start
    if str(data.get("model_id")) == "ssf2020":
        args += ["--seq-len", str(data.get("seq_len", 4)),
                 "--max-sequences", str(data.get("max_sequences", 64))]
        if not data.get("warm_start", True):
            args.append("--no-warm-start")
    if data.get("extra_args"):
        args += data["extra_args"].split()
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {
            "status": "started",
            "run_id": run_id,
            "pid": proc.pid,
            "config": data,
            "checkpoint": None,
            "metrics": None,
            "note": "训练后台运行中；完成后 metrics.json + checkpoint 见 /api/training/runs + /checkpoints。",
        }
    except FileNotFoundError:
        return {"status": "error", "run_id": run_id, "config": data, "checkpoint": None, "metrics": None,
                "note": f"训练脚本未找到: {script}"}


# ---- runs -------------------------------------------------------------- #

def _attach_ckpt_meta(run: dict) -> dict:
    """给 run 附 latest/best checkpoint 信息（从 checkpoints/<run>.ckpt.json 读，JSON 可含非展示字段）。"""
    rid = run.get("id", "")
    mf = os.path.join(CHECKPOINTS_DIR, f"{rid}.ckpt.json")
    meta = {}
    if os.path.isfile(mf):
        try:
            meta = json.loads(file_cached(mf, ttl=30.0) or "{}")
        except Exception:
            meta = {}
    run = dict(run)
    run["latest"] = meta.get("latest")
    run["best"] = meta.get("best")
    # checkpoint 文件存在性（best/latest 下载按钮据此显隐）
    run["has_latest"] = os.path.isfile(os.path.join(CHECKPOINTS_DIR, f"{rid}.pth"))
    run["has_best"] = os.path.isfile(os.path.join(CHECKPOINTS_DIR, f"{rid}.best.pth"))
    for key in ("model_id", "quality", "method", "dataset", "lambda", "size"):
        run[key] = meta.get(key)
    return run


# Fields stripped from /runs list responses to keep them lightweight.
# viz can be 6×epochs entries (7776+); test_metrics mirrors loss_series length.
# loss_series is kept — the frontend training curve overlay needs it.
_STRIP_FIELDS = ("test_metrics", "viz")


@router.get("/runs")
async def get_runs(model: str = None, dataset: str = None, status: str = None,
                   offset: int = None, limit: int = None, lite: bool = False):
    data = _load_metrics()
    runs = data.get("runs", [])
    if model:
        runs = [r for r in runs if r.get("model") == model]
    if dataset:
        runs = [r for r in runs if r.get("dataset") == dataset]
    if status:
        runs = [r for r in runs if r.get("status") == status]
    # lite: 额外 strip loss_series（EvalRun 只需 best/latest + model_id，不需曲线数据）
    strip = set(_STRIP_FIELDS) | ({"loss_series"} if lite else set())
    runs = [{k: v for k, v in _attach_ckpt_meta(r).items() if k not in strip}
            for r in runs]
    if offset is not None or limit is not None:
        off = offset or 0
        lim = limit or 50
        page = runs[off:off + lim]
        return {"total": len(runs), "offset": off, "limit": lim, "runs": page}
    return {"generated_at": data.get("generated_at"), "total": len(runs), "runs": runs}


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str):
    data = _load_metrics()
    for r in data.get("runs", []):
        if r.get("id") == run_id:
            return _attach_ckpt_meta(r)
    return {"detail": "Run not found"}, 404


# ---- checkpoints ------------------------------------------------------- #

@router.get("/checkpoints")
async def list_checkpoints():
    if not os.path.isdir(CHECKPOINTS_DIR):
        return {"checkpoints": []}
    out = []
    for fn in sorted(os.listdir(CHECKPOINTS_DIR)):
        full = os.path.join(CHECKPOINTS_DIR, fn)
        if not os.path.isfile(full) or fn.startswith('.') or not fn.endswith('.pth'):
            continue
        out.append({
            "id": os.path.splitext(fn)[0],
            "name": fn,
            "path": f"checkpoints/{fn}",
            "ext": ".pth",
            "size_bytes": os.path.getsize(full),
        })
    return {"checkpoints": out}


@router.get("/checkpoints/{checkpoint_id}")
async def get_checkpoint_detail(checkpoint_id: str):
    data = _load_metrics()
    for r in data.get("runs", []):
        cp = r.get("checkpoint_path", "")
        if cp and (checkpoint_id in cp or os.path.basename(cp).startswith(checkpoint_id)):
            return {"checkpoint": cp, "run": r}
    if os.path.isdir(CHECKPOINTS_DIR):
        for fn in os.listdir(CHECKPOINTS_DIR):
            if os.path.splitext(fn)[0] == checkpoint_id:
                return {"checkpoint": f"checkpoints/{fn}", "run": None}
    return {"detail": "Checkpoint not found"}, 404


# ---- outputs（按需服务 checkpoint/log，防穿越）------------------------ #

@router.get("/outputs")
async def list_outputs():
    if not os.path.isdir(TRAINING_OUTPUTS_DIR):
        return {"outputs": []}
    out = []
    for root, _, files in os.walk(TRAINING_OUTPUTS_DIR):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            if not os.path.isfile(full) or fn.startswith('.'):
                continue
            rel = os.path.relpath(full, TRAINING_OUTPUTS_DIR).replace(os.sep, '/')
            out.append({
                "name": fn, "path": rel,
                "ext": os.path.splitext(fn)[1].lower(),
                "size_bytes": os.path.getsize(full),
            })
    out.sort(key=lambda x: x["path"])
    return {"outputs": out}


@router.get("/outputs/{file_path:path}")
async def serve_output(file_path: str):
    safe = safe_resolve(TRAINING_OUTPUTS_DIR, file_path)
    if not safe or not os.path.isfile(safe):
        return {"detail": "Output not found"}, 404
    ext = os.path.splitext(safe)[1].lower()
    media = {
        ".pth": "application/octet-stream", ".log": "text/plain", ".json": "application/json",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    }.get(ext, "application/octet-stream")
    return FileResponse(safe, media_type=media, filename=os.path.basename(safe))
