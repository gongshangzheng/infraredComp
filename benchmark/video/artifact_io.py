"""Helpers to (re)load a ContourArtifact from its manifest.json on disk.

Lets stage 2 run independently of stage 1 (``--skip-extract``).
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import ContourArtifact


def load_artifact(contour_dir: str | Path) -> ContourArtifact:
    """Reconstruct a ContourArtifact from a datasets/contour/<name>/ manifest."""
    contour_dir = Path(contour_dir)
    manifest_path = contour_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {contour_dir}")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame_paths = [str(p) for p in sorted(contour_dir.glob("frame_*.png"))]
    return ContourArtifact(
        source_name=m.get("source_name", contour_dir.name),
        method=m.get("method", "unknown"),
        frames_dir=str(contour_dir),
        frame_paths=frame_paths,
        frame_count=m.get("frame_count", len(frame_paths)),
        fps=m.get("fps", 25.0),
        width=m.get("width", 0),
        height=m.get("height", 0),
        duration_s=m.get("duration_s", 0.0),
        manifest_path=str(manifest_path),
    )
