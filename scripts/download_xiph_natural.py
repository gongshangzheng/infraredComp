"""Download Xiph derf CIF natural video sequences (Y4M) for the contour baseline.

Lands 6 classic CIF Y4M clips under datasets/raw/xiph_cif/<name>.y4m so each can
be fed to the contour-video benchmark via --input (Y4M is in VIDEO_EXTS and ffmpeg
reads YUV4MPEG2 natively — no normalization needed).

Source:  https://media.xiph.org/video/derf/y4m/
License: Xiph.org derf test media — for testing/research (see Xiph terms).

Usage:
    uv run python scripts/download_xiph_natural.py            # idempotent
    uv run python scripts/download_xiph_natural.py --force    # re-fetch
    uv run python scripts/download_xiph_natural.py --dry-run  # plan only

Output:
    datasets/raw/xiph_cif/
        akiyo_cif.y4m ... mobile_cif.y4m
        manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Datasets 树位置可经 INFRACOMP_DATASETS_DIR 配置;默认 <repo>/datasets。
_DATASETS_DIR = Path(os.environ.get("INFRACOMP_DATASETS_DIR", str(PROJECT_ROOT / "datasets")))
sys.path.insert(0, str(PROJECT_ROOT))
from benchmark.video.config import raw_dir  # noqa: E402
OUT_DIR = raw_dir("xiph_cif")

BASE_URL = "https://media.xiph.org/video/derf/y4m"
SEQUENCES = ["akiyo_cif", "bus_cif", "city_cif", "flower_cif", "foreman_cif", "mobile_cif"]
DEFAULT_FPS = 25.0  # fallback only; real fps is probed per file

CITATION = (
    "Xiph.org derf collection - standard video test sequences. "
    "https://media.xiph.org/video/derf/"
)


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


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, **kw)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed (exit {exc.returncode}): {' '.join(cmd)}") from exc


def _download(name: str, dest: Path) -> None:
    url = f"{BASE_URL}/{name}.y4m"
    print(f"  > {url}")
    _run(["curl", "--ssl-no-revoke", "-fsSL", "--max-time", "300", "-o", str(dest), url])


def _probe(video: Path) -> tuple[int, int, float, int]:
    """Return (width, height, fps, frame_count) via ffprobe."""
    try:
        s = json.loads(
            subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                    "-of", "json", str(video),
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
    except Exception:  # noqa: BLE001
        return 0, 0, float(DEFAULT_FPS), 0


@dataclass(frozen=True)
class SeqInfo:
    name: str
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    size_bytes: int


def fetch_all(force: bool, dry_run: bool) -> list[SeqInfo]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("[dry-run] would download:")
        for name in SEQUENCES:
            print(f"  {BASE_URL}/{name}.y4m -> {OUT_DIR}/{name}.y4m")
        return []

    infos: list[SeqInfo] = []
    for name in SEQUENCES:
        dest = OUT_DIR / f"{name}.y4m"
        if dest.exists() and not force:
            print(f"* {name}.y4m exists ({dest.stat().st_size // 1024} KB), skip (--force to re-fetch)")
            w, h, fps, nbf = _probe(dest)
            infos.append(SeqInfo(name, dest, fps, nbf, w, h, dest.stat().st_size))
            continue
        print(f"* {name} <- {name}.y4m")
        _download(name, dest)
        w, h, fps, nbf = _probe(dest)
        infos.append(SeqInfo(name, dest, fps, nbf, w, h, dest.stat().st_size))
    return infos


def write_manifest(infos: list[SeqInfo]) -> None:
    if not infos:
        return
    manifest = {
        "dataset": "Xiph-CIF-natural",
        "source": f"{BASE_URL}/",
        "license": "Xiph.org derf test media (public test use)",
        "citation": CITATION,
        "format": "y4m (raw YUV4MPEG2)",
        "sequences": [
            {
                "id": i.name,
                "file": _rel_or_abs(i.path),
                "fps": i.fps,
                "frame_count": i.frame_count,
                "width": i.width,
                "height": i.height,
                "size_bytes": i.size_bytes,
            }
            for i in infos
        ],
    }
    p = OUT_DIR / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nmanifest -> {_rel_or_abs(p)}")


def print_baseline_hint() -> None:
    print("\nRun the natural-video baseline (all sequences):")
    print("  uv run python scripts/run_natural_baseline.py")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Xiph derf CIF natural video sequences (Y4M).")
    ap.add_argument("--force", action="store_true", help="re-download even if .y4m exists")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without downloading")
    args = ap.parse_args()

    _ensure_ffmpeg_on_path()
    if not shutil.which("ffprobe"):
        print("error: ffprobe not found (install system ffmpeg or `uv add static-ffmpeg`)", file=sys.stderr)
        return 2
    if not shutil.which("curl"):
        print("error: curl not found on PATH", file=sys.stderr)
        return 2

    print(f"output dir: {_rel_or_abs(OUT_DIR)}")
    infos = fetch_all(force=args.force, dry_run=args.dry_run)
    write_manifest(infos)
    print_baseline_hint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
