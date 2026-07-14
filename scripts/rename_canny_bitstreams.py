"""One-off: rename old canny bitstreams (pre-method-naming) to method-tagged names.

Old canny runs wrote `bitstreams/{seq}_{codec}_crf{N}.{ext}` (no method). After the
tag change (`{seq}_{method}_{codec}_crf{N}`), `_bitstream_for` no longer falls back
to the legacy name, so those files are orphaned. This script renames them in place
to `{seq}_canny_{codec}_crf{N}.{ext}` so canny runs resolve under the new naming.

Skips files already carrying a method token (canny/sobel/hed) before the codec.
Skips if the method-tagged target already exists (no overwrite). Bitstreams only —
recon dirs / decoded_sample are left untouched (different stored-path semantics).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BITSTREAMS = PROJECT_ROOT / "results" / "video" / "bitstreams"

METHODS = {"canny", "sobel", "hed"}
# codec ids from the registry (img-* contain hyphens, not underscores)
CODECS = {
    "x264", "x265", "svtav1", "vp9", "mpeg4", "ssf2020", "dcvc_rt",
    "img-bmshj2018-factorized", "img-bmshj2018-hyperprior",
    "img-cheng2020-anchor", "img-cheng2020-attn",
    "img-mbt2018", "img-mbt2018-mean", "img-ELIC",
}


def main() -> int:
    if not BITSTREAMS.is_dir():
        print(f"no bitstreams dir: {BITSTREAMS}")
        return 1
    renamed = skipped_tagged = skipped_exists = skipped_unknown = 0
    for p in sorted(BITSTREAMS.iterdir()):
        if not p.is_file():
            continue
        stem, ext = p.stem, p.suffix
        if "_crf" not in stem:
            continue
        base, _, crfpart = stem.rpartition("_crf")
        if not crfpart or not crfpart.isdigit():
            continue
        tokens = base.split("_")
        codec = tokens[-1]
        if codec not in CODECS:
            skipped_unknown += 1
            continue
        if len(tokens) >= 2 and tokens[-2] in METHODS:
            skipped_tagged += 1  # already method-tagged
            continue
        new_base = "_".join(tokens[:-1] + ["canny", codec])
        new_name = f"{new_base}_crf{crfpart}{ext}"
        new_path = BITSTREAMS / new_name
        if new_path.exists():
            skipped_exists += 1
            print(f"[skip-exists] {p.name} -> {new_name}")
            continue
        p.rename(new_path)
        renamed += 1
        print(f"[renamed] {p.name} -> {new_name}")
    print(f"\nrenamed={renamed} skipped(tagged)={skipped_tagged} "
          f"skipped(exists)={skipped_exists} skipped(unknown)={skipped_unknown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
