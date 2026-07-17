"""Download OTCBVS Dataset 03 (OSU Color-Thermal) thermal sequences and
organize them into the layout the contour-video benchmark consumes.

The benchmark's stage1 (`benchmark/video/stage1_extract.py`) reads from
``datasets/raw/``. This script lands one normalized ``.mp4`` per thermal
sequence under ``datasets/raw/osu_color_thermal/seqN.mp4`` so every sequence
can be fed in with the same ``--input`` shape.

Source
------
Page:   https://vcipl-okstate.org/pbvs/bench/Data/03/download.html
Files:  1a.zip .. 6a.zip  (the ``a`` archives are the THERMAL channel;
        ``b`` archives are the color channel and are deliberately skipped).

Each zip is treated opaquely — it may contain a single video file (avi/mov/…)
or a directory of bitmap frames (bmp/png/tif). Both are normalized to mp4:
  * video source  -> stream-copied into mp4 when the codec is mp4-compatible,
                     else re-encoded to yuv420p h264 (near-lossless, -crf 1).
  * frame source -> assembled with ffmpeg image2 into mp4 @ 25 fps
                     (DEFAULT_FRAME_FPS matches stage1's frames-dir default).

Usage
-----
    uv run python scripts/download_osu_color_thermal.py            # idempotent
    uv run python scripts/download_osu_color_thermal.py --force    # re-fetch
    uv run python scripts/download_osu_color_thermal.py --dry-run  # plan only

Output
------
    datasets/raw/osu_color_thermal/
        seq1.mp4 ... seq6.mp4
        manifest.json          # per-sequence source / fps / frame count / size

Then run stage1 per sequence, e.g.:
    uv run python -m benchmark.video \\
        --input datasets/raw/osu_color_thermal/seq1.mp4 \\
        --method canny --crfs 18,23,28,33
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants — single source of truth, no magic strings elsewhere.            #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Datasets 树位置可经 INFRACOMP_DATASETS_DIR 配置;默认 <repo>/datasets。
_DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(PROJECT_ROOT / "datasets")))
sys.path.insert(0, str(PROJECT_ROOT))
from benchmark.video.config import raw_dir  # noqa: E402
OUT_DIR = raw_dir("osu_color_thermal")


def _rel_or_abs(p: Path) -> str:
    """Path relative to repo root if inside it, else absolute (relocated datasets)."""
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _ensure_ffmpeg_on_path() -> None:
    """If system ffmpeg/ffprobe are missing, add the static-ffmpeg bundle to PATH."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    try:
        from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
        ffmpeg, _ffprobe = get_or_fetch_platform_executables_else_raise()
        os.environ["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # noqa: BLE001
        pass

BASE_URL = "https://vcipl-okstate.org/pbvs/bench/Data/03"
THERMAL_ZIPS = [f"{i}a.zip" for i in range(1, 7)]  # 1a..6a (thermal only)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_FPS = 25  # matches stage1_extract.DEFAULT_FRAME_FPS

# Per-source citation (OTCBVS requires acknowledgement on use).
CITATION = (
    "Davis, J. W., & Sharma, V. (2007). Background-subtraction using "
    "contour-based fusion of thermal and visible imagery. OTCBVS Dataset 03, "
    "OSU Color-Thermal Database. https://vcipl-okstate.org/pbvs/bench/"
)


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, raising with the failing command on error."""
    try:
        return subprocess.run(cmd, check=True, **kw)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed (exit {exc.returncode}): {' '.join(cmd)}") from exc


def _download(zip_name: str, dest: Path) -> None:
    url = f"{BASE_URL}/{zip_name}"
    print(f"  ↓ {url}")
    _run(["curl", "--ssl-no-revoke", "-fsSL", "--max-time", "180", "-o", str(dest), url])


def _find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file())


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


# --------------------------------------------------------------------------- #
# Core: turn one extracted zip into one normalized seqN.mp4                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeqInfo:
    index: int
    mp4: Path
    source_kind: str            # "video" | "frames"
    source_name: str
    fps: float
    frame_count: int
    width: int
    height: int
    size_bytes: int
    notes: str = ""


def _probe(video: Path) -> tuple[int, int, float, int]:
    """Return (width, height, fps, frame_count) via ffprobe."""
    try:
        s = json.loads(
            subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                    "-of", "json",
                    str(video),
                ]
            )
        )
        st = s["streams"][0]
        w, h = int(st.get("width", 0)), int(st.get("height", 0))
        rfr = st.get("r_frame_rate", "25/1")
        num, _, den = rfr.partition("/")
        fps = float(num) / float(den) if den and float(den) else DEFAULT_FPS
        nbf = int(st.get("nb_frames", 0) or 0)
        return w, h, fps, nbf
    except Exception:
        return 0, 0, float(DEFAULT_FPS), 0


def _ffmpeg_to_mp4(inputs: list[str], out: Path, extra: list[str]) -> None:
    """Run ffmpeg with given input args + extra filters, writing to ``out``."""
    cmd = ["ffmpeg", "-y", *inputs, *extra, "-pix_fmt", "yuv420p", str(out)]
    _run(cmd)


def _assemble_from_frames(images: list[Path], work: Path, out: Path) -> Path:
    """Rename frames to a zero-padded pattern, then ffmpeg image2 -> mp4.

    Renaming guarantees correct ordering regardless of the source naming
    (e.g. "1.bmp" / "10.bmp" / "2.bmp"), which ffmpeg's glob sort would misorder.
    """
    ext = images[0].suffix.lower()
    seq_dir = work / "frames"
    seq_dir.mkdir(exist_ok=True)
    for i, src in enumerate(images, start=1):
        shutil.copy2(src, seq_dir / f"frame_{i:06d}{ext}")
    pattern = str(seq_dir / f"frame_%06d{ext}")
    _ffmpeg_to_mp4(
        ["-framerate", str(DEFAULT_FPS), "-i", pattern], out,
        extra=["-c:v", "libx264", "-crf", "1"],
    )
    return out


def _copy_or_encode_video(src: Path, work: Path, out: Path) -> Path:
    """Stream-copy into mp4 when possible; fall back to a near-lossless re-encode."""
    tmp_copy = work / f"copy{src.suffix}"
    # Try stream copy into mp4 container first (no quality loss).
    try:
        _run(["ffmpeg", "-y", "-i", str(src), "-c", "copy", "-movflags", "+faststart",
               str(tmp_copy.with_suffix(".mp4"))])
        shutil.move(str(tmp_copy.with_suffix(".mp4")), str(out))
        return out
    except RuntimeError:
        pass
    # Fall back: re-encode (source is 8-bit thermal, yuv420p is lossless-ish).
    _ffmpeg_to_mp4(["-i", str(src)], out, extra=["-c:v", "libx264", "-crf", "1"])
    return out


def normalize_sequence(zip_path: Path, idx: int, work_root: Path) -> SeqInfo:
    """Extract ``zip_path`` and produce ``OUT_DIR/seq{idx}.mp4``."""
    out = OUT_DIR / f"seq{idx}.mp4"
    work = work_root / f"seq{idx}"
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work)

    videos = _find_videos(work)
    images = _find_images(work)

    if videos:
        src = max(videos, key=lambda p: p.stat().st_size)
        _copy_or_encode_video(src, work, out)
        kind, src_name = "video", src.name
    elif images:
        _assemble_from_frames(images, work, out)
        kind, src_name = "frames", f"{len(images)} images"
    else:
        raise RuntimeError(f"seq{idx}: no video or image frames found in {zip_path}")

    w, h, fps, nbf = _probe(out)
    size = out.stat().st_size if out.exists() else 0
    return SeqInfo(
        index=idx, mp4=out, source_kind=kind, source_name=src_name,
        fps=fps, frame_count=nbf, width=w, height=h, size_bytes=size,
    )


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def fetch_all(force: bool, dry_run: bool) -> list[SeqInfo]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("[dry-run] would download & normalize:")
        for z in THERMAL_ZIPS:
            print(f"  {BASE_URL}/{z} -> {OUT_DIR}/seq{z[0]}.mp4")
        return []

    infos: list[SeqInfo] = []
    with tempfile.TemporaryDirectory(prefix="osu_ct_") as tmp:
        tmp_path = Path(tmp)
        for idx, zip_name in enumerate(THERMAL_ZIPS, start=1):
            out = OUT_DIR / f"seq{idx}.mp4"
            if out.exists() and not force:
                print(f"* seq{idx}.mp4 exists ({out.stat().st_size // 1024} KB), skip (--force to re-fetch)")
                w, h, fps, nbf = _probe(out)
                infos.append(SeqInfo(idx, out, "cached", out.name, fps, nbf, w, h, out.stat().st_size))
                continue
            print(f"* seq{idx} <- {zip_name}")
            zip_path = tmp_path / zip_name
            _download(zip_name, zip_path)
            infos.append(normalize_sequence(zip_path, idx, tmp_path))
    return infos


def write_manifest(infos: list[SeqInfo]) -> None:
    if not infos:
        return
    manifest = {
        "dataset": "OSU Color-Thermal (OTCBVS Dataset 03)",
        "source": f"{BASE_URL}/download.html",
        "license": "Educational and research purposes only; must be acknowledged.",
        "citation": CITATION,
        "thermal_only": True,
        "sequences": [
            {
                "id": f"seq{i.index}",
                "file": _rel_or_abs(i.mp4),
                "source_kind": i.source_kind,
                "source_name": i.source_name,
                "fps": i.fps,
                "frame_count": i.frame_count,
                "width": i.width,
                "height": i.height,
                "size_bytes": i.size_bytes,
            }
            for i in infos
        ],
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest -> {_rel_or_abs(manifest_path)}")


def print_stage1_hint(infos: list[SeqInfo]) -> None:
    if not infos:
        return
    print("\nRun stage1 + stage2 per sequence:")
    for i in infos:
        rel = _rel_or_abs(i.mp4)
        print(f"  uv run python -m benchmark.video --input {rel} --method canny --crfs 18,23,28,33")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download & organize OSU Color-Thermal thermal sequences.")
    ap.add_argument("--force", action="store_true", help="re-download even if seqN.mp4 exists")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without downloading")
    args = ap.parse_args()

    _ensure_ffmpeg_on_path()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("error: ffmpeg/ffprobe not found (set INFRACOMP_FFMPEG_BIN, install "
              "system ffmpeg, or `uv add static-ffmpeg`)", file=sys.stderr)
        return 2
    if not shutil.which("curl"):
        print("error: curl not found on PATH", file=sys.stderr)
        return 2

    print(f"output dir: {_rel_or_abs(OUT_DIR)}")
    infos = fetch_all(force=args.force, dry_run=args.dry_run)
    write_manifest(infos)
    print_stage1_hint(infos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
