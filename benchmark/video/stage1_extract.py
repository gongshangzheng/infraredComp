"""Stage 1 — extract a lossless grayscale contour frame sequence from a raw video.

The output (a directory of PNG frames + manifest.json) is the ``ContourArtifact``
that stage 2 compresses and that quality metrics use as ground truth.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from . import config
from .data import ContourArtifact
from .extractors import build_extractor, list_extractors
from .ffmpeg_util import get_duration_seconds, get_stream_info, run_ffmpeg

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Default fps assumed when the input is a frame directory (no real container).
DEFAULT_FRAME_FPS = 25.0


def resolve_input(path: str | Path) -> tuple[str, Path]:
    """Classify an input as ('video', file) or ('frames', dir)."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        return "video", p
    if p.is_dir():
        return "frames", p
    raise FileNotFoundError(
        f"Input {path} is neither a video file ({sorted(VIDEO_EXTS)}) nor a directory of frames."
    )


def expand_inputs(input_args: list[str], *, skip_extract: bool = False) -> list[str]:
    """Flatten repeatable --input values into concrete source paths.

    - skip_extract=False: a directory is globbed for video files (VIDEO_EXTS);
      a file is kept as-is.
    - skip_extract=True: each entry is treated as ONE contour dir (no glob,
      since contour dirs hold PNGs, not videos).
    Raises FileNotFoundError on a missing path or an empty video directory.
    """
    out: list[str] = []
    for arg in input_args:
        p = Path(arg)
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {arg}")
        if p.is_dir() and not skip_extract:
            vids = sorted(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS
            )
            if not vids:
                raise FileNotFoundError(
                    f"No video files ({sorted(VIDEO_EXTS)}) in directory: {arg}"
                )
            out.extend(str(v) for v in vids)
        else:
            out.append(arg)
    return out


def _video_fps(path: Path) -> float:
    """Best-effort fps from ffprobe (r_frame_rate)."""
    try:
        stream = get_stream_info(path)
        rfr = stream.get("r_frame_rate", "0/1")
        num, _, den = rfr.partition("/")
        num_f = float(num) if num else 0.0
        den_f = float(den) if den else 1.0
        if den_f > 0 and num_f > 0:
            return num_f / den_f
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_FRAME_FPS


def demux_to_frames(video_path: Path, out_dir: Path, frames: int | None = None) -> list[Path]:
    """Demux a video to grayscale PNG frames via ffmpeg.

    Returns sorted list of produced frame paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.png")
    args = [
        "-y",
        "-i", str(video_path),
        "-pix_fmt", "gray",
        "-vsync", "0",
    ]
    if frames is not None:
        args += ["-vframes", str(frames)]
    args += [pattern]
    run_ffmpeg(args)
    return sorted(out_dir.glob("frame_*.png"))


def load_frame_sequence(frame_dir: Path) -> list[Path]:
    """Return sorted image paths from a directory of frames."""
    files = [f for f in sorted(frame_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")
    return files


def _read_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Failed to read frame: {path}")
    return arr


def extract_contour_video(
    raw_input: str | Path,
    method: str = "canny",
    out_dir: str | Path | None = None,
    frames: int | None = None,
    fps: float | None = None,
) -> ContourArtifact:
    """Extract a contour video from a raw video or frame directory.

    Parameters
    ----------
    raw_input : video file or frame directory
    method : registered extractor name (canny / sobel)
    out_dir : output dir (default datasets/contour/<source_name>/<method>)
    frames : cap frame count (useful for AV1-heavy stage-2 runs)
    fps : override fps (default: probed for video, 25 for frame dir)
    """
    if method not in list_extractors():
        raise KeyError(f"Unknown extractor '{method}'. Available: {list_extractors()}")

    src = Path(raw_input)
    kind, src_path = resolve_input(src)
    source_name = src_path.stem or src_path.name

    if out_dir is None:
        # 按方法分目录: datasets/contour/<source>/<method>/ —— 不同提取方法
        # (canny/sobel) 不互相覆盖；rmtree 只清该方法子目录，保留其它方法产物。
        out_dir = config.CONTOUR_DIR / source_name / method
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = build_extractor(method)

    # 1. Produce a raw grayscale frame sequence (in-memory paths)
    work_dir = out_dir / "_raw_frames"
    if kind == "video":
        demux_to_frames(src_path, work_dir, frames=frames)
        raw_frames = sorted(work_dir.glob("frame_*.png"))
        src_fps = fps if fps is not None else _video_fps(src_path)
        duration = get_duration_seconds(src_path)
    else:
        raw_frames = load_frame_sequence(src_path)
        if frames is not None:
            raw_frames = raw_frames[:frames]
        src_fps = fps if fps is not None else DEFAULT_FRAME_FPS
        duration = len(raw_frames) / src_fps if src_fps > 0 else 0.0

    if not raw_frames:
        raise RuntimeError(f"No frames produced from {src_path}")

    # 2. Extract edges per frame, write lossless grayscale PNG (the contour video)
    contour_paths: list[Path] = []
    width = height = 0
    for i, fp in enumerate(raw_frames):
        gray = _read_gray(fp)
        edges = extractor.extract(gray)
        if edges.ndim == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
        if i == 0:
            height, width = edges.shape
        cp = out_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(cp), edges)
        contour_paths.append(cp)

    # 3. Write manifest
    manifest = {
        "source_name": source_name,
        "method": method,
        "frame_count": len(contour_paths),
        "fps": src_fps,
        "width": width,
        "height": height,
        "frames_dir": str(out_dir),
        "duration_s": duration,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Clean raw frames (intermediate)
    shutil.rmtree(work_dir, ignore_errors=True)

    return ContourArtifact(
        source_name=source_name,
        method=method,
        frames_dir=str(out_dir),
        frame_paths=[str(p) for p in contour_paths],
        frame_count=len(contour_paths),
        fps=src_fps,
        width=width,
        height=height,
        manifest_path=str(manifest_path),
    )


def load_contour_frames(artifact: ContourArtifact) -> np.ndarray:
    """Load the lossless contour PNGs into an (N, H, W) uint8 array.

    Single read path for stage 2 ground truth and verify.
    """
    frames: list[np.ndarray] = []
    for p in artifact.frame_paths:
        arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"Failed to read contour frame: {p}")
        frames.append(arr)
    return np.stack(frames, axis=0)
