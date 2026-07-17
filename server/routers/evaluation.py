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
import sys
import time
from pathlib import Path

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
# Lazily-generated viewable mp4s for the speed page's 3-video cell:
#   source/      — original raw video (first N frames, matching the contour window)
#   contour_mp4/ — extracted edge video (from stage-1 contour PNGs)
# Both are served via /outputs/{path} like bitstreams/recon.
SOURCE_VIDEO_DIR = os.path.join(RESULTS_VIDEO_DIR, "source")
CONTOUR_VIDEO_DIR = os.path.join(RESULTS_VIDEO_DIR, "contour_mp4")
_VIDEO_CACHE: dict[tuple, str | None] = {}

# contour-video 的"模型"= 视频 codec（阶段2 压缩评测用）。
# 统一 codec 列表来自 benchmark 注册表（benchmark.video.codecs.catalog）—— 单一数据源，
# id 与注册表一致（不再有 bmshj2018-* vs img-bmshj2018-* 的错位）。传统 + 学习式 codec 同列。
def _codecs() -> list[dict]:
    """Unified codec catalog (traditional + learned), registry-derived."""
    from benchmark.video.codecs import catalog  # type: ignore
    return catalog()


def _traditional_codec_ids() -> list[str]:
    """Traditional ffmpeg codecs (CRF-sweepable). Used as /configs default."""
    return [c["id"] for c in _codecs() if c["kind"] == "codec"]


def _bench_python() -> str:
    """Python to run the benchmark subprocess in.

    The server usually runs under the `uv` venv (CPU-only torch). Learned codecs
    need a GPU torch env, so allow overriding via INFRACOMP_BENCH_PYTHON (e.g.
    point it at the conda env with CUDA). Falls back to sys.executable.
    """
    return os.environ.get("INFRACOMP_BENCH_PYTHON") or sys.executable


def _dataset_from_filename(name: str) -> str:
    """results.json -> 'default';其他 <dataset>.json -> 文件名 stem。"""
    stem = name[:-5] if name.endswith(".json") else name
    return "default" if stem == "results" else stem


_RESULTS_CACHE: dict = {}  # {mtime: float, ts: float, value: dict}


def _load_results() -> dict:
    """读 results/video/ 下所有 *.json(多数据集共存),聚合 runs。
    每条 run 带 dataset 字段(envelope dataset 优先,否则从文件名推断) +
    mode 字段(envelope mode 优先,否则按文件名/speed 推断:含 _speed -> speed,否则 formal)。

    5s TTL + mtime 感知缓存：文件没变就直接返回缓存的解析结果。
    """
    now = time.monotonic()
    # Compute the newest mtime among *.json files (0 if dir missing)
    newest_mtime = 0.0
    if os.path.isdir(RESULTS_VIDEO_DIR):
        for jf in Path(RESULTS_VIDEO_DIR).glob("*.json"):
            try:
                mt = jf.stat().st_mtime
                if mt > newest_mtime:
                    newest_mtime = mt
            except OSError:
                continue
    cached = _RESULTS_CACHE.get("value")
    if (cached is not None
            and _RESULTS_CACHE.get("mtime") == newest_mtime
            and (now - _RESULTS_CACHE.get("ts", 0)) < 5.0):
        return cached

    all_runs: list = []
    latest_gen = None
    if os.path.isdir(RESULTS_VIDEO_DIR):
        for jf in sorted(Path(RESULTS_VIDEO_DIR).glob("*.json")):
            content = read_file(str(jf))
            if not content:
                continue
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                runs = data.get("runs") or []
                ds = data.get("dataset") or _dataset_from_filename(jf.name)
                file_mode = data.get("mode")
                if file_mode is None:
                    file_mode = "speed" if "_speed" in jf.stem or "speed" in jf.name else "formal"
                gen = data.get("generated_at")
            elif isinstance(data, list):
                runs = data
                ds = _dataset_from_filename(jf.name)
                file_mode = "speed" if "_speed" in jf.stem or "speed" in jf.name else "formal"
                gen = None
            else:
                continue
            if gen and (latest_gen is None or gen > latest_gen):
                latest_gen = gen
            for r in runs:
                if isinstance(r, dict):
                    r.setdefault("dataset", ds)
                    r.setdefault("mode", file_mode)
                    r.pop("per_frame_psnr", None)  # strip: per-frame 数组占大头（300帧×2），前端聚合视图不需要
                    r.pop("per_frame_ssim", None)
                    all_runs.append(r)
    result = {"generated_at": latest_gen, "runs": all_runs}
    _RESULTS_CACHE.clear()
    _RESULTS_CACHE.update(value=result, mtime=newest_mtime, ts=now)
    return result


_BITSTREAM_INDEX: dict[str, str] = {}
_BITSTREAM_MTIME: float | None = None


def _bitstream_index() -> dict[str, str]:
    """扫一次 bitstreams/ (mtime-aware) -> {stem: 'bitstreams/{fn}'}。
    避免每 run os.listdir（3900× → 1×）。"""
    global _BITSTREAM_INDEX, _BITSTREAM_MTIME
    if not os.path.isdir(BITSTREAMS_DIR):
        _BITSTREAM_INDEX = {}; _BITSTREAM_MTIME = None
        return _BITSTREAM_INDEX
    mt = os.stat(BITSTREAMS_DIR).st_mtime
    if _BITSTREAM_INDEX and _BITSTREAM_MTIME == mt:
        return _BITSTREAM_INDEX
    idx = {}
    for fn in os.listdir(BITSTREAMS_DIR):
        if os.path.splitext(fn)[1].lower() in VIDEO_EXTS:
            idx[fn.rsplit(".", 1)[0]] = f"bitstreams/{fn}"
    _BITSTREAM_INDEX = idx; _BITSTREAM_MTIME = mt
    return idx


def _bitstream_for(run: dict) -> str | None:
    """给一条 run 找对应的压缩码流相对路径（bitstreams/{seq}_{method}_{codec}_crf{N}.{ext}）。

    只按含 method 的命名找 —— 名称必须与轮廓提取方法一致，绝不回退到其它方法
    （旧的无 method 命名不再复用，避免把 canny 重建当成 sobel/hed 展示）。
    """
    seq = run.get("sequence_name") or ""
    codec = run.get("codec") or ""
    crf = run.get("crf")
    method = run.get("method") or "canny"
    if not seq or not codec or crf is None:
        return None
    prefix = f"{seq}_{method}_{codec}_crf{crf}"
    idx = _bitstream_index()
    if prefix in idx:
        return idx[prefix]
    for stem, path in idx.items():
        if stem.startswith(prefix):
            return path
    return None


def _find_raw_video(seq: str) -> str | None:
    """Locate the original raw video for a sequence under datasets/raw/<dataset>/."""
    raw = os.path.join(DATASETS_DIR, "raw")
    if not os.path.isdir(raw):
        return None
    for ds in sorted(os.listdir(raw)):
        for ext in (".y4m", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"):
            cand = os.path.join(raw, ds, f"{seq}{ext}")
            if os.path.isfile(cand):
                return cand
    return None


def _contour_manifest(seq: str, method: str) -> dict:
    m = os.path.join(DATASETS_DIR, "contour", seq, method, "manifest.json")
    if not os.path.isfile(m):
        return {}
    try:
        return json.loads(read_file(m) or "{}")
    except json.JSONDecodeError:
        return {}


def _run_ffmpeg(args: list[str]) -> None:
    """ffmpeg via the benchmark's discovery (INFRACOMP_FFMPEG_BIN > PATH > static-ffmpeg)."""
    from benchmark.video.ffmpeg_util import run_ffmpeg  # type: ignore
    run_ffmpeg(args)


def _ensure_source_video(seq: str, dataset: str | None = None) -> str | None:
    """Lazy-generate a viewable mp4 of the ORIGINAL video, capped to the contour
    frame window so it lines up with the edge/recon clips in the 3-video cell.
    Returns relative path under OUTPUTS_DIR (e.g. 'source/akiyo_cif.mp4')."""
    key = ("source", seq, dataset)
    if key in _VIDEO_CACHE:
        return _VIDEO_CACHE[key]
    # BSDS 图片数据集:用 manifest source_images 映射原始 BSDS image,直接返 datasets 相对 path
    ds_lower = (dataset or "").lower()
    if "bsds" in ds_lower:
        split = "val" if "val" in ds_lower else ("test" if "test" in ds_lower else "val")
        mpath = os.path.join(DATASETS_DIR, "contour", f"bsds_{split}_gt", "manifest.json")
        if os.path.isfile(mpath):
            try:
                m = json.loads(read_file(mpath) or "{}")
                for si in m.get("source_images", []):
                    if si.get("stem") == seq:
                        img = os.path.join(DATASETS_DIR, "BSDS500", "images", split, si["image"])
                        if os.path.isfile(img):
                            rel = os.path.relpath(img, DATASETS_DIR).replace(os.sep, "/")
                            _VIDEO_CACHE[key] = rel
                            return rel
            except Exception:  # noqa: BLE001
                pass
    raw = _find_raw_video(seq)
    out_rel = f"source/{seq}.mp4"
    out = os.path.join(SOURCE_VIDEO_DIR, f"{seq}.mp4")
    if not raw or (os.path.isfile(out) and os.path.getsize(out) > 0):
        _VIDEO_CACHE[key] = out_rel if os.path.isfile(out) else None
        return _VIDEO_CACHE[key]
    # Match the evaluated frame window (contour frame_count); default 30.
    n = _contour_manifest(seq, "canny").get("frame_count") or 30
    os.makedirs(SOURCE_VIDEO_DIR, exist_ok=True)
    args = ["-y", "-i", raw, "-vframes", str(n),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out]
    try:
        _run_ffmpeg(args)
        _VIDEO_CACHE[key] = out_rel
    except Exception:  # noqa: BLE001
        _VIDEO_CACHE[key] = None
    return _VIDEO_CACHE[key]


def _ensure_contour_video(seq: str, method: str, dataset: str | None = None) -> str | None:
    """Lazy-generate a viewable lossy mp4 from the stage-1 lossless
    ``contour.mp4``. Returns relative path under OUTPUTS_DIR
    (e.g. 'contour_mp4/akiyo_cif_canny.mp4'). Returns None if the contour dir
    has no contour.mp4 (e.g. not yet extracted)."""
    key = ("contour", seq, method, dataset)
    if key in _VIDEO_CACHE:
        return _VIDEO_CACHE[key]
    # 图片数据集(BSDS/imagenet):method=gt → 直接返 datasets 下 gt png 相对路径,前端走 /datasets/{id}/media serve,不 copy
    if method == "gt":
        import glob
        ds_lower = (dataset or "").lower()
        gt_pattern = "bsds_val_gt" if "val" in ds_lower else ("bsds_test_gt" if "test" in ds_lower else "bsds_*_gt")
        for gt_dir in glob.glob(os.path.join(DATASETS_DIR, "contour", gt_pattern)):
            src = os.path.join(gt_dir, f"{seq}.png")
            if os.path.isfile(src):
                rel = os.path.relpath(src, DATASETS_DIR).replace(os.sep, "/")
                _VIDEO_CACHE[key] = rel
                return rel
    cdir = os.path.join(DATASETS_DIR, "contour", seq, method)
    out_rel = f"contour_mp4/{seq}_{method}.mp4"
    out = os.path.join(CONTOUR_VIDEO_DIR, f"{seq}_{method}.mp4")
    contour_mp4 = os.path.join(cdir, "contour.mp4")
    if not os.path.isfile(contour_mp4):
        _VIDEO_CACHE[key] = None
        return None
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        _VIDEO_CACHE[key] = out_rel
        return out_rel
    os.makedirs(CONTOUR_VIDEO_DIR, exist_ok=True)
    # lossless contour.mp4 -> lossy browser-playable H.264
    args = ["-y", "-i", contour_mp4,
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out]
    try:
        _run_ffmpeg(args)
        _VIDEO_CACHE[key] = out_rel
    except Exception:  # noqa: BLE001
        _VIDEO_CACHE[key] = None
    return _VIDEO_CACHE[key]


# ---- models（codec）-------------------------------------------------------- #

ROW_DEMO_DIR = os.path.join(RESULTS_VIDEO_DIR, "row_demo")


def _ensure_row_demo(seq: str, codec: str, crf, method: str) -> str | None:
    """拼一个"原始 | 轮廓 | 重建"横向三拼 mp4（formal 平均指标行右侧演示用）。
    三个输入:source/{seq}.mp4(原始)、contour_mp4/{seq}_{method}.mp4(轮廓)、
    bitstreams 重建(_bitstream_for)。三者必须同帧数同 fps 才能 hstack;取三者最短帧数对齐。
    返回相对 OUTPUTS_DIR 路径(row_demo/{seq}_{method}_{codec}_crf{crf}.mp4)。

    缓存 key / 输出文件名 / run 查找都带 method,避免 canny/sobel/hed 互相覆盖或串台。
    """
    method = method or "canny"
    key = ("row_demo", seq, method, codec, str(crf))
    if key in _VIDEO_CACHE:
        return _VIDEO_CACHE[key]
    out_rel = f"row_demo/{seq}_{method}_{codec}_crf{crf}.mp4"
    out = os.path.join(ROW_DEMO_DIR, f"{seq}_{method}_{codec}_crf{crf}.mp4")
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        _VIDEO_CACHE[key] = out_rel
        return out_rel
    src = _ensure_source_video(seq)
    con = _ensure_contour_video(seq, method)
    # 重建:从该 (seq, method, codec, crf) 的 run 找 bitstream。
    recon = None
    data = _load_results()
    for r in data.get("runs", []):
        if (r.get("sequence_name") == seq and r.get("codec") == codec
                and str(r.get("crf")) == str(crf)
                and (r.get("method") or "canny") == method):
            recon = _bitstream_for(r)
            break
    if not (src and con and recon):
        _VIDEO_CACHE[key] = None
        return None
    src_abs = os.path.join(OUTPUTS_DIR, src)
    con_abs = os.path.join(OUTPUTS_DIR, con)
    rec_abs = os.path.join(OUTPUTS_DIR, recon)
    if not all(os.path.isfile(p) for p in (src_abs, con_abs, rec_abs)):
        _VIDEO_CACHE[key] = None
        return None
    os.makedirs(ROW_DEMO_DIR, exist_ok=True)
    # hstack 三路;各路先 pad 到统一高 + 同 fps,取最短帧数对齐(避免长度不一致崩)。
    fps = _contour_manifest(seq, method).get("fps") or 25.0
    args = [
        "-y",
        "-i", src_abs, "-i", con_abs, "-i", rec_abs,
        "-filter_complex",
        f"[0:v]setpts=PTS-STARTPTS,scale=-2:ih,setsar=1[a];"
        f"[1:v]setpts=PTS-STARTPTS,scale=-2:ih,setsar=1[b];"
        f"[2:v]setpts=PTS-STARTPTS,scale=-2:ih,setsar=1[c];"
        f"[a][b][c]hstack=inputs=3[v]",
        "-map", "[v]",
        "-r", str(fps),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out,
    ]
    try:
        _run_ffmpeg(args)
        _VIDEO_CACHE[key] = out_rel
    except Exception:  # noqa: BLE001
        _VIDEO_CACHE[key] = None
    return _VIDEO_CACHE[key]


@router.get("/results/row_demo")
async def get_row_demo(codec: str, crf, method: str = "canny", dataset: str = None, mode: str = None):
    """给 formal 平均指标行直接流式返回三拼演示 mp4(原始|轮廓|重建 hstack)。
    前端 <video preload=none> 的 src 指向此端点点 play 才生成+流。"""
    data = _load_results()
    runs = data.get("runs", [])
    if mode:
        runs = [r for r in runs if r.get("mode") == mode]
    for r in runs:
        if r.get("codec") != codec or str(r.get("crf")) != str(crf):
            continue
        if dataset and (r.get("dataset") or r.get("sequence_name")) != dataset:
            continue
        # 按前端选的 method 过滤 —— 否则会取到同 (codec,crf,dataset) 的 canny run，
        # 把 canny 重建当成 sobel/hed 展示。method 默认 canny 兼容旧调用。
        if method and (r.get("method") or "canny") != method:
            continue
        seq = r.get("sequence_name") or ""
        if not seq:
            continue
        meth = r.get("method") or method
        rel = _ensure_row_demo(seq, codec, crf, meth)
        if rel:
            full = os.path.join(OUTPUTS_DIR, rel)
            return FileResponse(full, media_type="video/mp4",
                                filename=os.path.basename(full))
    return {"detail": "row demo not available"}, 404


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


@router.get("/codecs")
async def get_codecs():
    """列出所有 codec（传统 + 学习式），id 与 benchmark 注册表一致。模型即 codec。"""
    return _codecs()


@router.get("/codecs/{codec_id}")
async def get_codec_detail(codec_id: str):
    for c in _codecs():
        if c["id"] == codec_id:
            return c
    return {"detail": "Codec not found"}, 404


# ---- datasets（视频序列，按数据集家族分组）----------------------------------- #

DATASET_IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
}
DATASET_VIDEO_MIME = {
    ".mp4": "video/mp4", ".y4m": "application/octet-stream",
    ".avi": "video/x-msvideo", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
}


def _load_raw_datasets() -> list[dict]:
    """读取 datasets/raw/<dataset>/manifest.json，按真实数据集家族分组。"""
    out = []
    raw_dir = os.path.join(DATASETS_DIR, "raw")
    if not os.path.isdir(raw_dir):
        return out
    for ds_dir_name in sorted(os.listdir(raw_dir)):
        ds_dir = os.path.join(raw_dir, ds_dir_name)
        if not os.path.isdir(ds_dir):
            continue
        manifest_path = os.path.join(ds_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            manifest = json.loads(read_file(manifest_path) or "{}")
        except json.JSONDecodeError:
            manifest = {}
        sequences = []
        for seq in manifest.get("sequences") or []:
            seq_id = seq.get("id") or ""
            file_rel = seq.get("file") or ""
            # Normalize Windows backslash paths and resolve to actual filesystem
            if file_rel:
                # Try as relative path first (relative to DATASETS_DIR)
                norm_rel = file_rel.replace("\\", "/")
                if norm_rel.startswith("datasets/"):
                    # Strip leading datasets/ prefix — path is relative to repo root, resolve via DATASETS_DIR
                    candidate = os.path.join(DATASETS_DIR, norm_rel[len("datasets/"):])
                elif os.path.isabs(norm_rel):
                    candidate = norm_rel
                else:
                    candidate = os.path.join(ds_dir, norm_rel)
                file_path = candidate
            else:
                file_path = ""
            sequences.append({
                "id": seq_id,
                "name": os.path.basename(file_rel.replace("\\", "/")) if file_rel else seq_id,
                # Path relative to DATASETS_DIR (strip the redundant "datasets/"
                # prefix the manifest stores) so /datasets/{id}/media/{file} resolves.
                "file": (os.path.relpath(file_path, DATASETS_DIR).replace(os.sep, "/")
                         if file_path else ""),
                "fps": seq.get("fps", 0),
                "frame_count": seq.get("frame_count", 0),
                "width": seq.get("width", 0),
                "height": seq.get("height", 0),
                "size_bytes": seq.get("size_bytes", 0),
                "missing": not (file_path and os.path.isfile(file_path)),
                "contour": {},
            })
        out.append({
            "id": ds_dir_name,
            "name": manifest.get("dataset") or ds_dir_name,
            "kind": "raw",
            "source": manifest.get("source"),
            "license": manifest.get("license"),
            "citation": manifest.get("citation"),
            "format": manifest.get("format"),
            "description": manifest.get("description") or manifest.get("citation") or "",
            "sequences": sequences,
            "contour_methods": [],
        })
    return out


def _attach_contour(datasets: list[dict], *, with_previews: bool = False) -> None:
    """把 datasets/contour/<seq>/<method>/ 归属到对应 raw sequence 下。

    with_previews=True 时（仅详情页用）顺手生成可播放的原始/轮廓 mp4 路径
    （复用 _ensure_source_video / _ensure_contour_video），列表页保持轻量。
    """
    if not os.path.isdir(CONTOUR_DIR):
        return
    seq_map: dict[str, tuple[int, int]] = {}
    for di, ds in enumerate(datasets):
        for si, seq in enumerate(ds.get("sequences") or []):
            seq_map[seq.get("id", "")] = (di, si)

    standalone_contours = []
    for seq_dir_name in sorted(os.listdir(CONTOUR_DIR)):
        seq_dir = os.path.join(CONTOUR_DIR, seq_dir_name)
        if not os.path.isdir(seq_dir):
            continue
        for meth in sorted(os.listdir(seq_dir)):
            mdir = os.path.join(seq_dir, meth)
            if not os.path.isdir(mdir):
                continue
            manifest_path = os.path.join(mdir, "manifest.json")
            if not os.path.isfile(manifest_path):
                continue
            try:
                m = json.loads(read_file(manifest_path) or "{}")
            except json.JSONDecodeError:
                m = {}
            source_name = m.get("source_name") or seq_dir_name
            method = m.get("method") or meth
            frames_dir = m.get("frames_dir") or mdir
            info = {
                "method": method,
                "frame_count": m.get("frame_count", 0),
                "fps": m.get("fps", 0),
                "width": m.get("width", 0),
                "height": m.get("height", 0),
                "duration_s": m.get("duration_s", 0),
                "frames_dir": frames_dir,
                "sample_frames": _list_sample_frames(frames_dir),
                # 可播放轮廓视频（相对 OUTPUTS_DIR，经 /outputs/{path} 服务）
                "view_video": _ensure_contour_video(source_name, method) if with_previews else None,
            }
            if source_name in seq_map:
                di, si = seq_map[source_name]
                seq = datasets[di]["sequences"][si]
                seq.setdefault("contour", {})[method] = info
                if method not in datasets[di].get("contour_methods", []):
                    datasets[di].setdefault("contour_methods", []).append(method)
                # 每条序列只生成一次原始视频（多方法共享）。
                if with_previews and not seq.get("view_source"):
                    seq["view_source"] = _ensure_source_video(source_name)
            else:
                sc = {
                    "id": f"{seq_dir_name}/{meth}",
                    "name": f"{seq_dir_name} ({meth})",
                    "kind": "contour",
                    "source_name": source_name,
                    "contour": {method: info},
                }
                if with_previews:
                    sc["view_source"] = _ensure_source_video(source_name)
                standalone_contours.append(sc)
    if standalone_contours:
        datasets.append({
            "id": "contour_only",
            "name": "轮廓产物（独立）",
            "kind": "contour",
            "sequences": standalone_contours,
            "contour_methods": sorted({c["contour"][method]["method"] for c in standalone_contours for method in c["contour"]}),
        })


def _list_sample_frames(frames_dir: str, limit: int = 50) -> list[str]:
    """列出轮廓帧目录下的样例帧文件名（供前端 gallery 用）。"""
    if not frames_dir or not os.path.isdir(frames_dir):
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    try:
        return sorted([
            f for f in os.listdir(frames_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])[:limit]
    except OSError:
        return []


def _imagenet_image_datasets() -> list[dict]:
    """imagenet 图像数据集（val→speed run / test→formal），评测只用一部分图。

    imagenet 三 split 分工：train→训练，val→评测 speed run（图最少），test→评测 formal。
    图像是 parquet，不落地，评测按 sample_images 采样。id 用连字符单段
    （imagenet-val）以便 FastAPI 单段路由 + 前端 :id 路由匹配。
    """
    return [
        {
            "id": "imagenet-val", "name": "ImageNet val（检验 · speed run）",
            "kind": "image", "split": "val", "shards": 14, "sample_images": 200,
            "usage": "speed", "sequences": [], "contour_methods": [],
            "description": "speed run 用（图片最少的检验集），评测时只采样 200 张图，在线提边缘。",
        },
        {
            "id": "imagenet-test", "name": "ImageNet test（测试 · formal）",
            "kind": "image", "split": "test", "shards": 28, "sample_images": 500,
            "usage": "formal", "sequences": [], "contour_methods": [],
            "description": "正式评测用，只采样 500 张图，在线提边缘。",
        },
    ]


def _bsds_datasets() -> list[dict]:
    """BSDS500 val 轮廓 GT（已落地为 PNG），每张图视为 1 帧伪序列。

    需先运行 `python scripts/convert_bsds_gt.py --splits val` 生成
    datasets/contour/bsds_val_gt/frame_*.png + manifest.json。
    """
    return [{
        "id": "bsds-val",
        "name": "BSDS-val",
        "kind": "contour",
        "split": "val",
        "usage": "both",
        "sequences": [],
        "contour_methods": ["gt"],
        "description": "BSDS500 val 的 ground-truth 软边缘图，每张作为 1 帧伪序列跑视频 codec。",
    }]


# 已下线、不再注册到评测的数据集（raw 目录仍在但不展示）
_EVAL_HIDDEN_RAW = {"osu_color_thermal"}


@router.get("/datasets")
async def get_datasets():
    """列出数据集家族（Xiph-CIF-natural / imagenet val·test / BSDS val）及下属序列、轮廓方法。"""
    datasets = [d for d in _load_raw_datasets() if d["id"] not in _EVAL_HIDDEN_RAW]
    _attach_contour(datasets)
    datasets += _imagenet_image_datasets()
    datasets += _bsds_datasets()
    return datasets


@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(dataset_id: str):
    datasets = [d for d in _load_raw_datasets() if d["id"] not in _EVAL_HIDDEN_RAW]
    _attach_contour(datasets, with_previews=True)
    datasets += _imagenet_image_datasets()
    datasets += _bsds_datasets()
    ds = next((d for d in datasets if d["id"] == dataset_id), None)
    if not ds:
        return {"detail": "Dataset not found"}, 404
    return ds


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, method: str = "canny", n: int = 8):
    """imagenet 在线预览：从 parquet 采样 n 张，返回 原图↔边缘 对比（base64 PNG）。

    供数据集详情页 gallery（imagenet 不落地、无 PNG 文件，只能在线生成预览）。
    """
    if not dataset_id.startswith("imagenet-"):
        return {"previews": []}
    import base64
    import io
    import cv2
    import numpy as np
    from PIL import Image
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts.imagenet_contour_dataset import ImageNetContourDataset, split_from_dataset_id
    split = split_from_dataset_id(dataset_id)
    n = max(1, min(int(n), 16))
    ds = ImageNetContourDataset(split=split, method=method, max_images=n, size=128, shards=1)

    def _png_b64(arr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", arr)
        return f"data:image/png;base64,{base64.b64encode(buf).decode()}" if ok else ""

    previews: list[dict] = []
    for i in range(min(n, len(ds))):
        jpeg = ds.samples[i]
        if not jpeg:
            continue
        img = Image.open(io.BytesIO(jpeg)).convert("L")
        arr = np.array(img, dtype=np.uint8)
        arr = cv2.resize(arr, (128, 128))  # 与训练 size 对齐
        edges = ds._extractor.extract(arr)
        if edges.dtype != np.uint8:
            edges = edges.astype(np.uint8)
        previews.append({"original": _png_b64(arr), "edge": _png_b64(edges)})
    return {"dataset_id": dataset_id, "method": method, "split": split, "previews": previews}


@router.get("/datasets/{dataset_id}/media/{file_path:path}")
async def serve_dataset_media(dataset_id: str, file_path: str):
    """按需服务数据集内的图片/视频文件（raw 视频或轮廓帧），防路径穿越。"""
    safe = safe_resolve(DATASETS_DIR, file_path)
    if not safe or not os.path.isfile(safe):
        return {"detail": "Media not found"}, 404
    ext = os.path.splitext(safe)[1].lower()
    media = DATASET_IMAGE_MIME.get(ext) or DATASET_VIDEO_MIME.get(ext) or "application/octet-stream"
    return FileResponse(safe, media_type=media, filename=os.path.basename(safe))


@router.post("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    """触发数据集下载。目前仅支持 OSU Color-Thermal。"""
    if dataset_id != "osu_color_thermal":
        return {"detail": "Download not supported for this dataset"}, 400
    scripts_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(scripts_dir, "scripts", "download_osu_color_thermal.py")
    if not os.path.isfile(script):
        return {"detail": "Download script not found"}, 500
    try:
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {
            "status": "started",
            "pid": proc.pid,
            "script": script,
            "note": "OSU Color-Thermal 下载中，完成后刷新数据集列表。",
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "note": str(exc)}


# ---- configs（评测任务配置）----------------------------------------------- #

@router.get("/configs")
async def get_configs():
    """默认评测任务配置（codec × CRF × 提取器）。"""
    return [{
        "id": "default",
        "name": "默认 baseline",
        "codecs": _traditional_codec_ids(),
        "crfs": [18, 23, 28, 33],
        "methods": ["canny", "sobel", "hed", "pidinet", "yoloe26"],
        "description": "传统 codec × 4 CRF × 5 提取器（canny/sobel/hed/pidinet/yoloe26），全序列",
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
    """触发评测。统一 codec 列表（传统 + 学习式），/run 真正 Popen runner 跑起来。

    - xiph_cif 数据集 → scripts/run_all_subprocess.py（per-(seq,codec) 子进程隔离，
      支持学习式 codec 的段错误隔离 + per-codec qualities；写 results/video/xiph_cif.json）。
    - osu_color_thermal → scripts/run_osu_baseline.py（传统；学习式暂不支持）。
    - imagenet-* 图像数据集 → image benchmark CLI note（图像评测，非视频，仍走 CLI）。
    """
    dataset_id = data.get("dataset_id") or ""
    codecs = data.get("codecs") or []
    if isinstance(codecs, str):
        codecs = [c.strip() for c in codecs.split(",") if c.strip()]
    learned_ids = {c["id"] for c in _codecs() if c["kind"] != "codec"}
    has_learned = any(c in learned_ids for c in codecs)

    def _join_list(v):
        return ",".join(str(x) for x in v) if isinstance(v, list) else str(v)

    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")

    # imagenet 图像数据集：图像 benchmark（非视频），仍返回 CLI note。
    if dataset_id.startswith("imagenet-"):
        split = dataset_id.split("-", 1)[1] if "-" in dataset_id else "val"
        sample = 200 if split == "val" else (500 if split == "test" else 1000)
        model_id = data.get("model_id") or data.get("codec") or (codecs[0] if codecs else "")
        return {
            "status": "needs_image_runner",
            "config": data, "output_video": None, "metrics": None,
            "note": (f"imagenet {split} 图像评测（图像 benchmark，非视频）: "
                     f"python -m benchmark --learned {model_id or '<DL图像模型>'} "
                     f"--quality {data.get('quality', 1)} --imagenet-split {split} --max-images {sample}"),
        }

    mode = data.get("mode") or "speed"
    # 不截断帧：speed/formal 都用完整序列（speed 靠 --sequences 子集加速）。
    frames = None

    checkpoint = data.get("checkpoint")

    if dataset_id == "bsds-val":
        script = os.path.join(scripts_dir, "run_bsds_baseline.py")
        cmd = [_bench_python(), "-u", script, "--mode", mode]
        if codecs:
            cmd += ["--codecs", _join_list(codecs)]
        if data.get("crfs"):
            cmd += ["--crfs", _join_list(data["crfs"])]
        if mode == "speed" and data.get("sequences"):
            cmd += ["--sequences", _join_list(data["sequences"])]
        elif mode == "speed":
            # speed run 默认只跑前 50 张，快速出结果
            cmd += ["--max-images", "50"]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
    elif "osu_color_thermal" in dataset_id:
        if has_learned:
            return {"status": "error", "config": data, "output_video": None, "metrics": None,
                    "note": "osu 数据集暂不支持学习式 codec（无 runner）；用 xiph_cif。"}
        script = os.path.join(scripts_dir, "run_osu_baseline.py")
        cmd = [_bench_python(), script]
        if codecs:
            cmd += ["--codecs", _join_list(codecs)]
        if data.get("crfs"):
            cmd += ["--crfs", _join_list(data["crfs"])]
        if data.get("method"):
            cmd += ["--method", data["method"]]
        if mode == "speed" and data.get("sequences"):
            cmd += ["--sequences", _join_list(data["sequences"])]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
    else:
        # xiph_cif（及 fallback）：run_all_subprocess 支持传统 + 学习式，子进程隔离段错误。
        script = os.path.join(scripts_dir, "run_all_subprocess.py")
        cmd = [_bench_python(), "-u", script,
               "--mode", mode,
               "--codecs", _join_list(codecs or _traditional_codec_ids())]
        if frames is not None:
            cmd += ["--frames", str(frames)]
        if mode == "speed" and data.get("sequences"):
            cmd += ["--sequences", _join_list(data["sequences"])]
        if data.get("crfs"):
            cmd += ["--crfs", _join_list(data["crfs"])]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]

    env = {**os.environ, "NO_PROXY": "*", "no_proxy": "*", "PYTHONUTF8": "1"}
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        return {
            "status": "started",
            "pid": proc.pid,
            "config": data,
            "output_video": None,
            "metrics": None,
            "cmd": " ".join(cmd),
            "note": f"后台运行中({os.path.basename(script)}, {len(codecs)} codec);"
                    f"完成后结果见 /evaluation/results,输出码流见 /evaluation/outputs。",
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
async def get_results(model: str = None, dataset: str = None, metric: str = None, mode: str = None,
                      offset: int = None, limit: int = None):
    """列 contour-video 评测结果（每条附 output_video 供按需播放）。
    mode=formal/speed 过滤(formal→xiph_cif.json,speed→xiph_cif_speed.json 分文件)。
    offset/limit 分页：提供时返回 {total, offset, limit, runs}，不提供时返回裸数组（向后兼容）。"""
    data = _load_results()
    runs = data.get("runs", [])
    if mode:
        runs = [r for r in runs if r.get("mode") == mode]
    # Per-request dedup caches: avoid repeated os.listdir / ffmpeg probes
    # for the same (seq, method) across different codec/crf runs.
    src_cache: dict[str, str | None] = {}
    con_cache: dict[str, str | None] = {}
    out = []
    for r in runs:
        vid = _bitstream_for(r)
        row = dict(r)
        row["output_video"] = vid
        seq = r.get("sequence_name") or ""
        method = r.get("method") or "canny"
        if seq:
            src_key = f"{seq}/{r.get('dataset')}"
            if src_key not in src_cache:
                src_cache[src_key] = _ensure_source_video(seq, r.get("dataset"))
            row["original_video"] = src_cache[src_key]
            con_key = f"{seq}/{method}/{r.get('dataset')}"
            if con_key not in con_cache:
                con_cache[con_key] = _ensure_contour_video(seq, method, r.get("dataset"))
            row["contour_video"] = con_cache[con_key]
        else:
            row["original_video"] = None
            row["contour_video"] = None
        row["model_name"] = r.get("codec")
        row["dataset_name"] = r.get("dataset") or r.get("sequence_name")
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
    # Pagination
    if offset is not None or limit is not None:
        off = offset or 0
        lim = limit or 50
        page = out[off:off + lim]
        return {"total": len(out), "offset": off, "limit": lim, "runs": page}
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


@router.get("/results/aggregate")
async def aggregate_results(dataset: str = None, method: str = None, mode: str = None):
    """per-(codec,crf) 平均(跨所有 seq),formal test 视图用。复用 aggregate_by_codec_crf。"""
    from types import SimpleNamespace
    from benchmark.video.aggregate import aggregate_by_codec_crf
    data = _load_results()
    runs = data.get("runs", [])
    if mode:
        runs = [r for r in runs if r.get("mode") == mode]
    if dataset:
        runs = [r for r in runs if (r.get("dataset") or r.get("sequence_name")) == dataset]
    if method:
        runs = [r for r in runs if r.get("method") == method]
    fields = ["codec", "codec_family", "crf", "psnr", "ssim", "bitrate_kbps", "bpp",
              "compression_ratio", "enc_fps", "dec_fps", "temporal_metric", "compressed_bytes"]
    objs = [SimpleNamespace(**{k: r.get(k) for k in fields}) for r in runs]
    return aggregate_by_codec_crf(objs)


@router.get("/results/{result_id}")
async def get_result_detail(result_id: str):
    for r in await get_results():
        if r.get("id") == result_id:
            return r
    return {"detail": "Result not found"}, 404


# ---- outputs（按需服务输出视频/码流，防穿越）------------------------------- #

@router.get("/outputs")
async def list_outputs(offset: int = None, limit: int = None):
    """列 OUTPUTS_DIR 下可查看的输出文件（bitstreams 压缩码流 + recon 重建帧）。
    offset/limit 分页：提供时返回 {total, offset, limit, outputs}。"""
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
    if offset is not None or limit is not None:
        off = offset or 0
        lim = limit or 100
        page = out[off:off + lim]
        return {"total": len(out), "offset": off, "limit": lim, "outputs": page}
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
