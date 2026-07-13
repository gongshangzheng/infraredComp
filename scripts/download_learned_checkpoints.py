"""Download pretrained checkpoints for the learned video compression models
used by the contour-video benchmark.

The benchmark loads these weights eagerly via CompressAI's pretrained zoo, and
`benchmark/learned.py:_load_model` + `benchmark/video/codecs/ssf2020.py` both
expect the files under ``~/.cache/torch/hub/checkpoints/`` (the standard
``torch.hub`` cache). Downloading them up front with this script makes the
benchmark reproducible and offline-friendly; it is the explicit-curl twin of
CompressAI's ``load_state_dict_from_url``.

Models
------
  * ssf2020  — CompressAI's Scale-Space Flow video model (qualities 1-9, mse).
               URLs come straight from the installed CompressAI package:
               ``compressai.zoo.video.model_urls["ssf2020"]["mse"]``.
  * dcvc-rt  — PLACEHOLDER. The repo / checkpoint source is not vendored yet.
               ``download_dcvc_rt()`` is a stub that raises NotImplementedError
               so the wiring is in place once DCVC-RT lands.

Usage
-----
    python3 scripts/download_learned_checkpoints.py            # idempotent (ssf2020 q1-9)
    python3 scripts/download_learned_checkpoints.py --force   # re-download
    python3 scripts/download_learned_checkpoints.py --dry-run  # plan only
    python3 scripts/download_learned_checkpoints.py --model ssf2020   # filter
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants — single source of truth, no magic strings elsewhere.            #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CompressAI + torch.hub both cache under ~/.cache/torch/hub/checkpoints/.
# This matches benchmark/learned.py:_load_model and benchmark/video/codecs/ssf2020.py.
CACHE_DIR = Path(os.environ.get(
    "TORCH_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "torch"),
)) / "hub" / "checkpoints"

QUALITIES = range(1, 10)  # ssf2020 quality levels 1..9


def _rel_or_abs(p: Path) -> str:
    """Path relative to repo root if inside it, else absolute (relocated home/cache)."""
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, raising with the failing command on error."""
    try:
        return subprocess.run(cmd, check=True, **kw)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"command failed (exit {exc.returncode}): {' '.join(cmd)}"
        ) from exc


def _download(url: str, dest: Path) -> None:
    """curl a single URL to ``dest`` (curl chosen for control + project convention)."""
    print(f"  ↓ {url}")
    _run([
        "curl", "--ssl-no-revoke", "-fsSL",
        "--max-time", "600",
        "-o", str(dest), url,
    ])


def _ssf2020_urls() -> dict[int, str]:
    """Pull the ssf2020 mse URLs from the installed CompressAI package."""
    from compressai.zoo.video import model_urls  # verified import path
    try:
        mse_urls = model_urls["ssf2020"]["mse"]
    except KeyError as exc:
        raise RuntimeError(
            "CompressAI ssf2020 mse URLs not found in compressai.zoo.video.model_urls"
        ) from exc
    return {q: mse_urls[q] for q in QUALITIES}


# --------------------------------------------------------------------------- #
# ssf2020                                                                      #
# --------------------------------------------------------------------------- #
def download_ssf2020(force: bool, dry_run: bool) -> list[tuple[int, str, str, str]]:
    """Download ssf2020 mse checkpoints q1..q9 into the torch.hub cache.

    Returns a list of (quality, filename, url, status) where status is one of
    "cached", "downloaded", "would-download", "skip".
    """
    urls = _ssf2020_urls()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[tuple[int, str, str, str]] = []

    for q in QUALITIES:
        url = urls[q]
        fname = url.rsplit("/", 1)[-1]
        dest = CACHE_DIR / fname
        if dry_run:
            status = "would-download" if (force or not dest.exists()) else "cached"
            results.append((q, fname, url, status))
            continue
        if dest.exists() and not force:
            size_kb = dest.stat().st_size // 1024
            print(f"* q{q} {fname} exists ({size_kb} KB), skip (--force to re-fetch)")
            results.append((q, fname, url, "skip"))
            continue
        print(f"* q{q} <- {fname}")
        try:
            _download(url, dest)
        except RuntimeError as exc:
            print(f"  ! download failed: {exc}", file=sys.stderr)
            if dest.exists():
                dest.unlink()  # no half files
            results.append((q, fname, url, "failed"))
            continue
        results.append((q, fname, url, "downloaded"))

    return results


# --------------------------------------------------------------------------- #
# dcvc-rt (PLACEHOLDER)                                                        #
# --------------------------------------------------------------------------- #
def download_dcvc_rt(force: bool, dry_run: bool) -> list[tuple[int, str, str, str]]:
    """Download DCVC-RT checkpoint.

    PLACEHOLDER — the DCVC-RT repo / checkpoint source is not vendored yet.
    Once vendored, add the checkpoint URL/source here (do NOT fabricate one).
    """
    raise NotImplementedError(
        "DCVC-RT: repo TBD — once vendored, add its checkpoint URL/source here. "
        "The benchmark does not yet ship DCVC-RT wiring; this stub exists so the "
        "download entry point is ready when the repo is integrated."
    )


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
DISPATCH = {
    "ssf2020": download_ssf2020,
    "dcvc-rt": download_dcvc_rt,
}


def _print_summary(results: list[tuple[int, str, str, str]]) -> None:
    if not results:
        return
    cached = [r for r in results if r[3] in ("cached", "skip")]
    downloaded = [r for r in results if r[3] == "downloaded"]
    failed = [r for r in results if r[3] == "failed"]
    would = [r for r in results if r[3] == "would-download"]
    if cached:
        print("\nalready cached:")
        for q, fname, _url, _st in cached:
            print(f"  q{q}  {_rel_or_abs(CACHE_DIR / fname)}")
    if downloaded:
        print("\ndownloaded:")
        for q, fname, _url, _st in downloaded:
            print(f"  q{q}  {_rel_or_abs(CACHE_DIR / fname)}")
    if would:
        print("\nwould download:")
        for q, fname, url, _st in would:
            print(f"  q{q}  {url} -> {_rel_or_abs(CACHE_DIR / fname)}")
    if failed:
        print("\nfailed:")
        for q, fname, url, _st in failed:
            print(f"  q{q}  {url}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download pretrained checkpoints for learned video compression models.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="re-download even if checkpoint already cached",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="show the plan without downloading",
    )
    ap.add_argument(
        "--model", choices=sorted(DISPATCH.keys()), default="ssf2020",
        help="which model to fetch (default: ssf2020)",
    )
    args = ap.parse_args()

    if not shutil.which("curl"):
        print("error: curl not found on PATH", file=sys.stderr)
        return 2

    print(f"cache dir: {_rel_or_abs(CACHE_DIR)}")
    fn = DISPATCH[args.model]
    try:
        results = fn(force=args.force, dry_run=args.dry_run)
    except NotImplementedError as exc:
        print(f"\n[skip] {exc}", file=sys.stderr)
        return 0  # placeholder models are non-fatal

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
