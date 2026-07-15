"""Pre-download CompressAI pretrained weights into the torch hub cache.

The benchmark's ``_load_model`` pre-checks the cache and raises if a weight is
missing, so image-model weights must be present before running img-* codecs.
This script fetches exactly the qualities the benchmark uses (see
``benchmark/learned.py`` LEARNED_MODELS and ``codecs/learned_image.py``).

Uses urllib with NO_PROXY so it bypasses any dead system proxy.
"""
from __future__ import annotations

import os
import sys
import urllib.request

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

from compressai.zoo.image import model_urls  # noqa: E402

# (model_name, [qualities]) — mirrors benchmark/learned.py + learned_image.py
NEEDED = [
    ("bmshj2018-factorized", [1, 4, 8]),
    ("bmshj2018-hyperprior", [1, 4, 8]),
    ("mbt2018", [1, 4, 8]),
    ("mbt2018-mean", [1, 4, 8]),
    ("cheng2020-anchor", [1, 4, 6]),
    ("cheng2020-attn", [1, 4, 6]),
]

CACHE_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")
os.makedirs(CACHE_DIR, exist_ok=True)


def main() -> int:
    failures = 0
    for name, quals in NEEDED:
        for q in quals:
            url = model_urls[name]["mse"][q]
            fname = url.rsplit("/", 1)[-1]
            dst = os.path.join(CACHE_DIR, fname)
            if os.path.isfile(dst) and os.path.getsize(dst) > 0:
                print(f"[skip] {name}-q{q} already cached ({os.path.getsize(dst)} B)")
                continue
            print(f"[get ] {name}-q{q} <- {url}")
            try:
                urllib.request.urlretrieve(url, dst)
                print(f"       ok {os.path.getsize(dst)} B")
            except Exception as e:  # noqa: BLE001
                print(f"       FAIL: {e}", file=sys.stderr)
                failures += 1
    print(f"\nDone. failures={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
