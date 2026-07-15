"""Helpers to (re)load a ContourArtifact from its manifest.json on disk.

Lets stage 2 run independently of stage 1 (``--skip-extract``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import ContourArtifact


def load_artifact(contour_dir: str | Path) -> ContourArtifact:
    """Reconstruct a ContourArtifact from a datasets/contour/<name>/ manifest.

    The persistent artifact is the lossless contour.mp4 (``video_path``); the
    per-frame PNGs are not kept. ``frame_paths`` stays empty — stage 2
    materializes transient frames from the video at run time.
    """
    contour_dir = Path(contour_dir)
    manifest_path = contour_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {contour_dir}")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    video_path = m.get("video_path") or str(contour_dir / "contour.mp4")
    return ContourArtifact(
        source_name=m.get("source_name", contour_dir.name),
        method=m.get("method", "unknown"),
        frames_dir=str(contour_dir),
        frame_paths=[],  # PNGs not kept; stage 2 materializes from contour.mp4
        frame_count=m.get("frame_count", 0),
        fps=m.get("fps", 25.0),
        width=m.get("width", 0),
        height=m.get("height", 0),
        duration_s=m.get("duration_s", 0.0),
        manifest_path=str(manifest_path),
        video_path=video_path,
    )
