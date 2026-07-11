"""评测体系路由 — infraredComp contour-video 适配版。

复用上游 ProjFlow 的 evaluation 契约（models/datasets/configs/run/results/outputs），
把领域数据源接到 contour-video：
  models    = 视频 codec（x264/x265/svtav1/vp9，来自 benchmark/video/codecs/ 注册表）
  datasets  = 视频序列（datasets/raw/ 原始 + datasets/contour/ 阶段1 产物）
  configs   = 评测任务默认配置（codec / crf / 提取器）
  results   = results/video/results.json（每条 run 增 output_video 字段指向 bitstreams/）
  outputs   = results/video/ 下文件（bitstreams 压缩码流 + recon 重建帧），按需 FileResponse
  run       = 触发 scripts/run_osu_baseline.py（异步起，立即返回 started）

输出视频按需服务：前端 <video preload="none"> 仅在用户点开时才请求 /outputs/{path}。
"""
import os
import json
import subprocess

from fastapi import APIRouter, Body
from fastapi.responses import FileResponse

from server.config import (
    DATASETS_DIR, CONTOUR_DIR, OUTPUTS_DIR, RESULTS_VIDEO_DIR, RESULTS_VIDEO_JSON,
)
from server.utils.file_utils import read_file, safe_resolve

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v"}
VIDEO_MIME = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".mkv": "video/x-matroska",
    ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".m4v": "video/x-m4v",
}
BITSTREAMS_DIR = os.path.join(RESULTS_VIDEO_DIR, "bitstreams")
RECON_DIR = os.path.join(RESULTS_VIDEO_DIR, "recon")

# contour-video 的"模型"= 视频 codec（阶段2 压缩评测用）
DEFAULT_CODECS = [
    {"id": "x264", "name": "x264 (H.264/AVC)", "type": "h264", "params": {"encoder": "libx264", "ext": ".mp4"}, "description": "最通用基线 codec"},
    {"id": "x265", "name": "x265 (HEVC)", "type": "hevc", "params": {"encoder": "libx265", "ext": ".mp4"}, "description": "高压缩比现代 codec"},
    {"id": "svtav1", "name": "SVT-AV1", "type": "av1", "params": {"encoder": "libsvtav1", "ext": ".mp4"}, "description": "新一代 royalty-free，较慢"},
    {"id": "vp9", "name": "VP9", "type": "vp9", "params": {"encoder": "libvpx-vp9", "ext": ".webm"}, "description": "Google 开源 codec"},
]


def _load_results() -> dict:
    """读 results/video/results.json。返回 {generated_at, runs: []}。"""
    content = read_file(RESULTS_VIDEO_JSON)
    if not content:
        return {"generated_at": None, "runs": []}
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "runs" in data:
            return data
        return {"generated_at": None, "runs": data if isinstance(data, list) else []}
    except json.JSONDecodeError:
        return {"generated_at": None, "runs": []}


def _bitstream_for(run: dict) -> str | None:
    """给一条 run 找对应的压缩码流相对路径（bitstreams/{seq}_{codec}_crf{N}.{ext}）。"""
    seq = run.get("sequence_name") or ""
    codec = run.get("codec") or ""
    crf = run.get("crf")
    if not seq or not codec or crf is None:
        return None
    prefix = f"{seq}_{codec}_crf{crf}"
    if not os.path.isdir(BITSTREAMS_DIR):
        return None
    for fn in os.listdir(BITSTREAMS_DIR):
        if fn.startswith(prefix) and os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
            return f"bitstreams/{fn}"
    return None


# ---- models（codec）-------------------------------------------------------- #

@router.get("/methods")
async def get_methods():
    """列出 results 中出现过的轮廓提取方法（canny/sobel/...）+ 已知提取器。"""
    data = _load_results()
    seen = []
    for r in data.get("runs", []):
        m = r.get("method")
        if m and m not in seen:
            seen.append(m)
    # 兜底：从 benchmark/video/extractors 注册表补
    try:
        from benchmark.video.extractors import list_extractors  # type: ignore
        for m in list_extractors():
            if m not in seen:
                seen.append(m)
    except Exception:  # noqa: BLE001
        pass
    return {"methods": seen}


@router.get("/models")
async def get_models():
    """列出可用 codec（contour-video 的"模型"= 阶段2 压缩 codec）。"""
    return DEFAULT_CODECS


@router.get("/models/{model_id}")
async def get_model_detail(model_id: str):
    for m in DEFAULT_CODECS:
        if m["id"] == model_id:
            return m
    return {"detail": "Model not found"}, 404


# ---- datasets（视频序列）--------------------------------------------------- #

@router.get("/datasets")
async def get_datasets():
    """列出原始视频序列 + 阶段1 轮廓产物。"""
    out = []
    raw_dir = os.path.join(DATASETS_DIR, "raw")
    if os.path.isdir(raw_dir):
        for d in sorted(os.listdir(raw_dir)):
            sub = os.path.join(raw_dir, d)
            if not os.path.isdir(sub):
                continue
            for fn in sorted(os.listdir(sub)):
                if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
                    out.append({"id": f"{d}/{fn}", "name": fn, "kind": "raw", "path": f"raw/{d}/{fn}"})
    # 阶段1 轮廓产物（按方法分目录: <source>/<method>/manifest.json；兼容旧扁平布局）
    if os.path.isdir(CONTOUR_DIR):
        for d in sorted(os.listdir(CONTOUR_DIR)):
            src_dir = os.path.join(CONTOUR_DIR, d)
            if not os.path.isdir(src_dir):
                continue
            # 找该 source 下所有方法的 manifest（新布局 src/<method>/manifest.json）
            method_dirs = [s for s in sorted(os.listdir(src_dir))
                           if os.path.isdir(os.path.join(src_dir, s)) and
                           os.path.isfile(os.path.join(src_dir, s, "manifest.json"))]
            # 兼容旧布局: src/manifest.json（单方法）
            if not method_dirs and os.path.isfile(os.path.join(src_dir, "manifest.json")):
                method_dirs = [""]
            for meth in method_dirs:
                mdir = os.path.join(src_dir, meth) if meth else src_dir
                try:
                    m = json.loads(read_file(os.path.join(mdir, "manifest.json")) or "{}")
                except json.JSONDecodeError:
                    m = {}
                mid = f"contour/{d}/{meth}" if meth else f"contour/{d}"
                out.append({
                    "id": mid, "name": (f"{d} ({meth})" if meth else d), "kind": "contour",
                    "method": m.get("method", meth or "unknown"),
                    "frame_count": m.get("frame_count", 0), "fps": m.get("fps", 0),
                    "width": m.get("width", 0), "height": m.get("height", 0),
                })
    return out


@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(dataset_id: str):
    for d in await get_datasets():
        if d["id"] == dataset_id:
            return d
    return {"detail": "Dataset not found"}, 404


# ---- configs（评测任务配置）----------------------------------------------- #

@router.get("/configs")
async def get_configs():
    """默认评测任务配置（codec × CRF × 提取器）。"""
    return [{
        "id": "default",
        "name": "默认 baseline",
        "codecs": [c["id"] for c in DEFAULT_CODECS],
        "crfs": [18, 23, 28, 33],
        "methods": ["canny", "sobel"],
        "description": "4 codec × 4 CRF × 2 提取器，全序列",
    }]


@router.get("/configs/{config_id}")
async def get_config_detail(config_id: str):
    for c in await get_configs():
        if c["id"] == config_id:
            return c
    return {"detail": "Config not found"}, 404


# ---- run（触发 baseline）-------------------------------------------------- #

@router.post("/run")
async def run_evaluation(data: dict = Body(...)):
    """异步触发 scripts/run_osu_baseline.py，立即返回 started。

    结果不在此返回——写入 results/video/results.json 后由 /results 读取，
    输出码流在 /outputs，前端按需播放。
    """
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "run_osu_baseline.py")
    try:
        proc = subprocess.Popen(["python3", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {
            "status": "started",
            "pid": proc.pid,
            "config": data,
            "output_video": None,
            "metrics": None,
            "note": "baseline 后台运行中；完成后结果见 /evaluation/results，输出码流见 /evaluation/outputs。",
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "config": data,
            "output_video": None,
            "metrics": None,
            "note": f"runner 脚本未找到: {script}",
        }


# ---- results -------------------------------------------------------------- #

@router.get("/results")
async def get_results(model: str = None, dataset: str = None, metric: str = None):
    """列 contour-video 评测结果（每条附 output_video 供按需播放）。"""
    data = _load_results()
    runs = data.get("runs", [])
    # 给每条 run 附 output_video（若码流存在）
    out = []
    for r in runs:
        vid = _bitstream_for(r)
        row = dict(r)
        row["output_video"] = vid
        # 兼容上游契约：model_name / dataset_name / metrics
        row["model_name"] = r.get("codec")
        row["dataset_name"] = r.get("sequence_name")
        row["metrics"] = {
            "psnr": r.get("psnr"), "ssim": r.get("ssim"),
            "bitrate_kbps": r.get("bitrate_kbps"), "bpp": r.get("bpp"),
            "compression_ratio": r.get("compression_ratio"),
            "enc_fps": r.get("enc_fps"), "dec_fps": r.get("dec_fps"),
        }
        row["timestamp"] = data.get("generated_at")
        out.append(row)
    if model:
        out = [r for r in out if r.get("model_name") == model]
    if dataset:
        out = [r for r in out if r.get("dataset_name") == dataset]
    return out


@router.get("/results/compare")
async def compare_results(models: str = None, datasets: str = None):
    runs = await get_results()
    ml = models.split(",") if models else None
    dl = datasets.split(",") if datasets else None
    if ml:
        runs = [r for r in runs if r.get("model_name") in ml]
    if dl:
        runs = [r for r in runs if r.get("dataset_name") in dl]
    grouped: dict[str, list] = {}
    for r in runs:
        grouped.setdefault(r.get("model_name", "?"), []).append(r)
    return {"codecs": grouped}


@router.get("/results/{result_id}")
async def get_result_detail(result_id: str):
    for r in await get_results():
        if r.get("id") == result_id:
            return r
    return {"detail": "Result not found"}, 404


# ---- outputs（按需服务输出视频/码流，防穿越）------------------------------- #

@router.get("/outputs")
async def list_outputs():
    """列 OUTPUTS_DIR 下可查看的输出文件（bitstreams 压缩码流 + recon 重建帧）。"""
    if not os.path.isdir(OUTPUTS_DIR):
        return {"outputs": []}
    out = []
    for root, _, files in os.walk(OUTPUTS_DIR):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, OUTPUTS_DIR).replace(os.sep, "/")
            ext = os.path.splitext(fn)[1].lower()
            out.append({
                "name": fn, "path": rel, "ext": ext,
                "is_video": ext in VIDEO_MIME,
                "size_bytes": os.path.getsize(full),
            })
    out.sort(key=lambda x: x["path"])
    return {"outputs": out}


@router.get("/outputs/{file_path:path}")
async def serve_output(file_path: str):
    """按需流式服务一个输出文件。路径经 safe_resolve 必须位于 OUTPUTS_DIR 内。

    前端 <video preload="none"> 仅在用户点开时才请求此端点（按需加载）。
    """
    safe = safe_resolve(OUTPUTS_DIR, file_path)
    if not safe or not os.path.isfile(safe):
        return {"detail": "Output not found"}, 404
    ext = os.path.splitext(safe)[1].lower()
    media = VIDEO_MIME.get(ext, "application/octet-stream")
    return FileResponse(safe, media_type=media, filename=os.path.basename(safe))
