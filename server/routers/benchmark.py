"""Benchmark router — serves persisted contour-video results (read-only).

Reads results/video/results.json (written by `python -m benchmark.video`).
Execution (the runner) is decoupled from this reporting layer — never depends
on the runner being live (ProjFlow boundary).
"""
import json
import os
from typing import Optional

from fastapi import APIRouter

from server.config import CONTOUR_DIR, RESULTS_VIDEO_JSON
from server.cache import file_cached

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


def _load_results() -> dict:
    """Read the persisted results JSON. Returns {generated_at, runs: []} if absent."""
    content = file_cached(RESULTS_VIDEO_JSON, ttl=5.0)
    if not content:
        return {"generated_at": None, "runs": []}
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "runs" in data:
            return data
        return {"generated_at": None, "runs": data if isinstance(data, list) else []}
    except json.JSONDecodeError:
        return {"generated_at": None, "runs": []}


@router.get("/results")
async def get_results(
    codec: Optional[str] = None,
    sequence: Optional[str] = None,
    crf: Optional[int] = None,
):
    """List benchmark runs, optionally filtered by codec / sequence / crf."""
    data = _load_results()
    runs = data.get("runs", [])
    if codec:
        runs = [r for r in runs if r.get("codec") == codec]
    if sequence:
        runs = [r for r in runs if r.get("sequence_name") == sequence]
    if crf is not None:
        runs = [r for r in runs if r.get("crf") == crf]
    return {"generated_at": data.get("generated_at"), "total": len(runs), "runs": runs}


@router.get("/results/compare")
async def compare_results(codecs: Optional[str] = None, sequences: Optional[str] = None):
    """Group runs by codec for comparison. ?codecs=x264,x265&sequences=foo"""
    data = _load_results()
    codec_list = codecs.split(",") if codecs else None
    seq_list = sequences.split(",") if sequences else None
    runs = data.get("runs", [])
    if codec_list:
        runs = [r for r in runs if r.get("codec") in codec_list]
    if seq_list:
        runs = [r for r in runs if r.get("sequence_name") in seq_list]
    grouped: dict[str, list] = {}
    for r in runs:
        grouped.setdefault(r.get("codec", "?"), []).append(r)
    return {"codecs": grouped}


@router.get("/runs")
async def list_runs():
    """List available contour videos (datasets/contour/<source>/<method>/ manifest).

    兼容旧扁平布局 datasets/contour/<source>/manifest.json（单方法）。
    """
    if not os.path.isdir(CONTOUR_DIR):
        return {"runs": []}
    runs = []
    for d in sorted(os.listdir(CONTOUR_DIR)):
        full = os.path.join(CONTOUR_DIR, d)
        if not os.path.isdir(full):
            continue
        # 新布局：<source>/<method>/manifest.json；旧布局：<source>/manifest.json
        has_top_manifest = os.path.isfile(os.path.join(full, "manifest.json"))
        sub_dirs = [s for s in sorted(os.listdir(full))
                    if os.path.isdir(os.path.join(full, s)) and
                    os.path.isfile(os.path.join(full, s, "manifest.json"))]
        targets = [os.path.join(full, s) for s in sub_dirs] or (
            [full] if has_top_manifest else []
        )
        for t in targets:
            manifest = os.path.join(t, "manifest.json")
            try:
                m = json.loads(file_cached(manifest, ttl=30.0) or "{}")
            except json.JSONDecodeError:
                m = {}
            method = m.get("method", os.path.basename(t) if t != full else "unknown")
            runs.append({
                "name": f"{d}/{method}" if t != full else d,
                "source": d,
                "method": method,
                "frame_count": m.get("frame_count", 0),
                "fps": m.get("fps", 0),
                "width": m.get("width", 0),
                "height": m.get("height", 0),
                "duration_s": m.get("duration_s", 0),
            })
    return {"runs": runs}


@router.post("/run")
async def run_benchmark():
    """Stub: execution is via CLI (`python -m benchmark.video`). Decoupled by design."""
    return {
        "status": "pending",
        "note": "评测执行请走 CLI: uv run python -m benchmark.video --input ...",
    }
