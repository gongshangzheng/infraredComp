"""ffmpeg / ffprobe subprocess helpers.

The project has no Python video binding (no av / decord / imageio-ffmpeg), so
all video demux / encode / decode goes through the system ffmpeg subprocess.
"""

import json
import shutil
import subprocess
from pathlib import Path

_FFMPEG_CANDIDATES = ("ffmpeg", "/opt/homebrew/bin/ffmpeg")
_FFPROBE_CANDIDATES = ("ffprobe", "/opt/homebrew/bin/ffprobe")


def find_ffmpeg() -> str:
    """Locate the ffmpeg binary. Raises FileNotFoundError if missing."""
    for c in _FFMPEG_CANDIDATES:
        path = shutil.which(c)
        if path:
            return path
    raise FileNotFoundError(
        "ffmpeg not found. Install it (e.g. `brew install ffmpeg`) with "
        "libx264/libx265/libsvtav1/libvpx support."
    )


def find_ffprobe() -> str:
    """Locate the ffprobe binary."""
    for c in _FFPROBE_CANDIDATES:
        path = shutil.which(c)
        if path:
            return path
    raise FileNotFoundError("ffprobe not found. Install ffmpeg (it ships ffprobe).")


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    """Run ffmpeg with the given args. Raises on non-zero exit.

    stderr is captured and included on failure for easier debugging.
    """
    cmd = [find_ffmpeg(), *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (code {proc.returncode}):\n"
            f"command: {' '.join(cmd)}\nstderr:\n{proc.stderr[-2000:]}"
        )
    return proc


def probe(path: str | Path) -> dict:
    """Return ffprobe JSON (format + streams) for a media file."""
    cmd = [
        find_ffprobe(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr}")
    return json.loads(proc.stdout)


def get_stream_info(path: str | Path) -> dict:
    """Return the first video stream of a media file."""
    data = probe(path)
    streams = data.get("streams", [])
    for s in streams:
        if s.get("codec_type") == "video":
            return s
    raise RuntimeError(f"No video stream found in {path}")


def get_duration_seconds(path: str | Path) -> float:
    """Return duration in seconds from ffprobe (0.0 if unknown)."""
    data = probe(path)
    fmt = data.get("format", {})
    try:
        return float(fmt.get("duration", 0.0))
    except (TypeError, ValueError):
        return 0.0
