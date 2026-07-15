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
from .ffmpeg_util import (
    demux_to_frames,
    find_ffmpeg,
    get_duration_seconds,
    get_stream_info,
    run_ffmpeg,
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".y4m"}
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
    skip_if_exists: bool = False,
) -> ContourArtifact:
    """Extract a contour video from a raw video or frame directory.

    Parameters
    ----------
    raw_input : video file or frame directory
    method : registered extractor name (canny / sobel)
    out_dir : output dir (default datasets/contour/<source_name>/<method>)
    frames : cap frame count (useful for AV1-heavy stage-2 runs)
    fps : override fps (default: probed for video, 25 for frame dir)
    skip_if_exists : if True, reuse the existing contour dir when its
        manifest.json already records this method with a valid contour.mp4
        (training-flow idempotency; evaluation keeps the default re-build).

    The persistent artifact is a lossless contour.mp4; the per-frame PNGs are
    deleted after stitching (stage 2 materializes transient frames from the
    video at run time).
    """
    if method not in list_extractors():
        raise KeyError(f"Unknown extractor '{method}'. Available: {list_extractors()}")

    src = Path(raw_input)
    kind, src_path = resolve_input(src)
    source_name = src_path.stem or src_path.name

    if out_dir is None:
        # 按方法分目录: datasets/contour/<source>/<method>/ —— 不同提取方法
        # (canny/sobel) 不互相覆盖。重建走 temp+原子替换，失败不毁既有产物。
        out_dir = config.CONTOUR_DIR / source_name / method
    out_dir = Path(out_dir)

    # 训练流程幂等：产物已存在且 method 匹配、contour.mp4 在 → 直接复用。
    if skip_if_exists:
        manifest_path = out_dir / "manifest.json"
        if manifest_path.is_file():
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                m = {}
            vp = m.get("video_path") or str(out_dir / "contour.mp4")
            if (m.get("method") == method and (m.get("frame_count") or 0) > 0
                    and Path(vp).is_file()):
                return ContourArtifact(
                    source_name=m.get("source_name", source_name),
                    method=method,
                    frames_dir=str(out_dir),
                    frame_paths=[],  # PNGs not kept; stage 2 materializes from video
                    frame_count=m.get("frame_count", 0),
                    fps=m.get("fps", 0.0),
                    width=m.get("width", 0),
                    height=m.get("height", 0),
                    manifest_path=str(manifest_path),
                    video_path=vp,
                )

    # Build into a temp sibling dir, then atomically swap on success. A
    # failed/interrupted re-extraction never wipes the existing contour frames:
    # the old dir stays until the new one is fully written and ready.
    work_out = out_dir.parent / (out_dir.name + ".tmp")
    if work_out.exists():
        shutil.rmtree(work_out)
    work_out.mkdir(parents=True, exist_ok=True)

    extractor = build_extractor(method)

    # 1. fps / duration (for the contour-mp4 stitch + manifest). The extractor
    #    now owns the video→frames split (extract_video), so stage1 no longer
    #    demuxes/loops here — it just assembles the contour.mp4 from edge frames.
    if kind == "video":
        src_fps = fps if fps is not None else _video_fps(src_path)
        duration = get_duration_seconds(src_path)
    else:
        _dir_frames = load_frame_sequence(src_path)
        _n = len(_dir_frames[:frames] if frames is not None else _dir_frames)
        src_fps = fps if fps is not None else DEFAULT_FRAME_FPS
        duration = _n / src_fps if src_fps > 0 else 0.0

    # 2. Extractor owns decoding + (per-frame or native-video) edge extraction.
    #    Each model decides gray-vs-color; returns a list of uint8 HxW edge frames.
    edges = extractor.extract_video(src_path, fps=src_fps, frames=frames)
    if not edges:
        raise RuntimeError(f"No frames produced from {src_path}")

    # 3. Write edge frames to lossless grayscale PNGs (intermediate for the stitch).
    contour_paths: list[Path] = []
    width = height = 0
    for i, e in enumerate(edges):
        if e.ndim == 3:
            e = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
        if i == 0:
            height, width = e.shape
        cp = work_out / f"frame_{i:06d}.png"
        cv2.imwrite(str(cp), e)
        contour_paths.append(cp)

    # 3. Stitch the contour PNGs into a LOSSLESS video (the persistent product).
    #    -qp 0 = lossless; yuv420p Y plane is exact for grayscale edges.
    #    pad odd dims to even (libx264 yuv420p requirement); stage 2 crops back.
    contour_mp4 = work_out / "contour.mp4"
    run_ffmpeg([
        "-y",
        "-framerate", str(src_fps),
        "-i", str(work_out / "frame_%06d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p",
        str(contour_mp4),
    ])

    # 4. Delete the contour frame PNGs — not kept on disk; stage 2 materializes
    #    transient frames from contour.mp4 at run time.
    for cp in contour_paths:
        try:
            cp.unlink()
        except OSError:
            pass

    # 5. Write manifest (frames_dir + video_path point at the final out_dir)
    final_video = out_dir / "contour.mp4"
    manifest = {
        "source_name": source_name,
        "method": method,
        "frame_count": len(contour_paths),
        "fps": src_fps,
        "width": width,
        "height": height,
        "frames_dir": str(out_dir),
        "video_path": str(final_video),
        "duration_s": duration,
    }
    work_manifest = work_out / "manifest.json"
    work_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Swap: replace the existing dir only after the new one is fully ready.
    # (The raw-frame decode temp is owned + cleaned by extractor.extract_video.)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    work_out.rename(out_dir)

    return ContourArtifact(
        source_name=source_name,
        method=method,
        frames_dir=str(out_dir),
        frame_paths=[],  # PNGs deleted; stage 2 materializes from contour.mp4
        frame_count=len(contour_paths),
        fps=src_fps,
        width=width,
        height=height,
        manifest_path=str(out_dir / "manifest.json"),
        video_path=str(final_video),
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


def load_contour_video_frames(artifact: ContourArtifact) -> np.ndarray:
    """Decode the lossless contour video into an (N, H, W) uint8 array.

    Replaces load_contour_frames for the video-based artifact (PNGs deleted).
    ffmpeg pipes raw grayscale bytes; we reshape at the padded dims (lossless
    stitch pads odd dims to even) and crop back to (height, width).
    """
    import subprocess
    if not artifact.video_path:
        raise RuntimeError("artifact has no video_path; cannot load contour video frames")
    w, h = artifact.width, artifact.height
    if w <= 0 or h <= 0:
        raise RuntimeError("artifact missing width/height to crop decoded frames")
    args = [find_ffmpeg(), "-i", artifact.video_path,
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for {artifact.video_path}: "
            f"{proc.stderr.decode(errors='ignore')[:300]}")
    pw, ph = w + (w % 2), h + (h % 2)  # even dims after lossless stitch's pad
    frame_bytes = pw * ph
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        raise RuntimeError(f"no frames decoded from {artifact.video_path} "
                            f"(got {len(proc.stdout)} bytes, need {frame_bytes}/frame)")
    arr = np.frombuffer(proc.stdout[:n * frame_bytes], dtype=np.uint8).reshape(n, ph, pw)
    return arr[:, :h, :w]  # crop pad back to original dims
