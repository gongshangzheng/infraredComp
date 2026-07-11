"""Reproducibility-metadata builder for the contour-video benchmark envelope.

Produces a dict merged into results.json alongside {generated_at, runs}, carrying
the codec/crf/extractor/git-sha context needed to reproduce or compare a run.
Git lookup never raises (timeout + broad except) so a benchmark run in any
environment still succeeds.
"""

from __future__ import annotations

import subprocess

from . import config


def _git(args: list[str], timeout: int = 5) -> str:
    """Run a git command in the repo root; return stdout or '' on any failure."""
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(config.BASE_DIR),
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def git_sha() -> str:
    """HEAD commit sha, or '' if git is unavailable / not a repo."""
    return _git(["rev-parse", "HEAD"])


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes (False on any error)."""
    return bool(_git(["status", "--porcelain"]))


def build_metadata(
    *,
    inputs: list[str],
    codecs: list[str],
    crfs: list[int],
    method: str,
    frame_cap: int | None,
    runner: str,
    dataset: str | None = None,
) -> dict:
    """Compose the reproducibility envelope for results.json. Never raises."""
    sha = git_sha()
    meta: dict = {
        "benchmark": "contour-video",
        "method": method,
        "codecs": list(codecs),
        "crfs": list(crfs),
        "frame_cap": frame_cap,
        "inputs": list(inputs),
        "sequence_count": len(inputs),
        "runner": runner,
        "git_sha": sha,
        "git_dirty": git_dirty() if sha else False,
    }
    if dataset:
        meta["dataset"] = dataset
    return meta
