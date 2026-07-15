"""Fetch the PiDiNet checkpoint into the vendored model dir.

PiDiNet ships its weights *committed in the upstream repo*
(``trained_models/table5_pidinet.pth``, ~3 MB) — no OneDrive/Google Drive. So
this script shallow-clones ``hellozhuo/pidinet`` to a temp dir, copies the pth
into ``third_party/pidinet/``, and removes the temp clone. Idempotent (skips if
already present + non-empty). Uses NO_PROXY to bypass any dead system proxy.

The output path matches ``extractors/pidinet.py``'s ``DEFAULT_WEIGHTS`` exactly
(``third_party/pidinet/table5_pidinet.pth``) — no manual rename needed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST = PROJECT_ROOT / "third_party" / "pidinet" / "table5_pidinet.pth"
REPO = "https://github.com/hellozhuo/pidinet.git"
SRC_IN_REPO = "trained_models/table5_pidinet.pth"


def main() -> int:
    if DEST.is_file() and DEST.stat().st_size > 0:
        print(f"[skip] {DEST} already present ({DEST.stat().st_size} B)")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pidinet_"))
    try:
        print(f"[clone] {REPO} -> {tmp} (shallow)")
        r = subprocess.run(
            ["git", "clone", "--depth", "1", REPO, str(tmp)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"[fail] git clone: {r.stderr[-1000:]}", file=sys.stderr)
            return 1
        src = tmp / SRC_IN_REPO
        if not src.is_file():
            print(f"[fail] {SRC_IN_REPO} not found in clone", file=sys.stderr)
            return 1
        shutil.copy2(src, DEST)
        print(f"[ok] {DEST} ({DEST.stat().st_size} B)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
