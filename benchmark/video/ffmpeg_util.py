"""ffmpeg / ffprobe subprocess helpers.

The project has no Python video binding (no av / decord / imageio-ffmpeg), so
all video demux / encode / decode goes through the system ffmpeg subprocess.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

_FFMPEG_CANDIDATES = ("ffmpeg", "/opt/homebrew/bin/ffmpeg")
_FFPROBE_CANDIDATES = ("ffprobe", "/opt/homebrew/bin/ffprobe")


def _from_env(env_var: str, name: str) -> str | None:
    """Resolve a binary from an env var pointing at the exe or its directory."""
    val = os.getenv(env_var)
    if not val:
        return None
    p = Path(val)
    if p.is_dir():
        for cand in (p / name, p / f"{name}.exe"):
            if cand.is_file():
                return str(cand)
    elif p.is_file() and p.name.lower() in (name, f"{name}.exe"):
        return str(p)
    return None


def _from_static_ffmpeg(name: str) -> str | None:
    """Use the pip `static-ffmpeg` bundle if installed (ships ffmpeg + ffprobe)."""
    try:
        from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
        ffmpeg, ffprobe = get_or_fetch_platform_executables_else_raise()
        return ffmpeg if name == "ffmpeg" else ffprobe
    except Exception:  # noqa: BLE001
        return None


def find_ffmpeg() -> str:
    """Locate ffmpeg: INFRACOMP_FFMPEG_BIN env > PATH > static_ffmpeg bundle."""
    path = _from_env("INFRACOMP_FFMPEG_BIN", "ffmpeg")
    if path:
        return path
    for c in _FFMPEG_CANDIDATES:
        found = shutil.which(c)
        if found:
            return found
    path = _from_static_ffmpeg("ffmpeg")
    if path:
        return path
    raise FileNotFoundError(
        "ffmpeg not found. Set INFRACOMP_FFMPEG_BIN, install system ffmpeg, or "
        "`uv add static-ffmpeg` (bundled ffmpeg + ffprobe)."
    )


def find_ffprobe() -> str:
    """Locate ffprobe: INFRACOMP_FFMPEG_BIN env > PATH > static_ffmpeg bundle."""
    path = _from_env("INFRACOMP_FFMPEG_BIN", "ffprobe")
    if path:
        return path
    for c in _FFPROBE_CANDIDATES:
        found = shutil.which(c)
        if found:
            return found
    path = _from_static_ffmpeg("ffprobe")
    if path:
        return path
    raise FileNotFoundError(
        "ffprobe not found. Set INFRACOMP_FFMPEG_BIN, install system ffmpeg, or "
        "`uv add static-ffmpeg`."
    )


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
