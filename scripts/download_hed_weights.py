"""Pre-cache the HED (Holistically-Nested Edge Detection) weights.

Downloads ``deploy.prototxt`` + ``hed_pretrained.caffemodel`` — the original
`s9xie/hed` Berkeley release — into the shared cache dir
``~/.cache/infraredcomp/hed/`` so ``extractors/hed.py`` can load them via
``cv2.dnn.readNetFromCaffe`` (no Caffe install).

Mirrors: the model has been re-hosted in many places; we try a list per file and
use the first that works. Uses urllib with NO_PROXY so it bypasses any dead
system proxy (see memory: Google Drive is NOT reachable here, so Drive-hosted
mirrors are omitted — only GitHub raw / release assets).
"""

from __future__ import annotations

import os
import sys
import urllib.request

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

# Mirrors tried in order; first reachable wins. Add known-good ones as found.
PROTOTXT_MIRRORS = [
    "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_deploy.prototxt",
]
# caffemodel (~56 MB). Plain (non-LFS) GitHub raw serves up to 100 MB, so these
# work if the host repo stores the binary as a normal blob.
CAFFEMODEL_MIRRORS = [
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed_pretrained.caffemodel",
]

CACHE_DIR = os.path.expanduser("~/.cache/infraredcomp/hed")
EXPECTED_CAFFEMODEL_BYTES = 56 * 1024 * 1024  # ~56 MB; sanity floor


def _fetch(url: str, dst: str) -> bool:
    """Download ``url`` to ``dst``. Return True on (plausible) success."""
    try:
        print(f"  [try ] {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "infraredcomp/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
        size = os.path.getsize(dst)
        print(f"  [ok  ] {size} B")
        return size > 0
    except Exception as e:  # noqa: BLE001
        print(f"  [fail] {e}", file=sys.stderr)
        try:
            if os.path.isfile(dst):
                os.remove(dst)
        except OSError:
            pass
        return False


def _download_file(mirrors: list[str], dst: str, *, binary: bool) -> bool:
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {dst} already present ({os.path.getsize(dst)} B)")
        return True
    for url in mirrors:
        if _fetch(url, dst):
            # Sanity-check the payload shape before accepting.
            if binary:
                if os.path.getsize(dst) < EXPECTED_CAFFEMODEL_BYTES // 4:
                    print(f"  [warn] suspiciously small for a caffemodel, trying next mirror",
                          file=sys.stderr)
                    os.remove(dst)
                    continue
            else:
                with open(dst, "rb") as f:
                    head = f.read(64).lstrip()
                if not head.lower().startswith(b"name:") and b"layer" not in head.lower():
                    print(f"  [warn] doesn't look like a prototxt, trying next mirror",
                          file=sys.stderr)
                    os.remove(dst)
                    continue
            return True
    return False


def main() -> int:
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[cache] {CACHE_DIR}")

    ok_pt = _download_file(
        PROTOTXT_MIRRORS, os.path.join(CACHE_DIR, "deploy.prototxt"), binary=False
    )
    ok_cm = _download_file(
        CAFFEMODEL_MIRRORS,
        os.path.join(CACHE_DIR, "hed_pretrained.caffemodel"),
        binary=True,
    )
    print()
    if ok_pt and ok_cm:
        print("Done. HED ready — extractors/hed.py can load it.")
        return 0
    print("FAILED to fetch one or both HED files. Add a reachable mirror to this "
          "script's *_MIRRORS list and re-run.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
