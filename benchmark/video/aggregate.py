"""Aggregation for video benchmark results.

ONE shared aggregator (snake_case) consumed by both the matplotlib charts and
the HTML report — avoids the Title_Case/snake_case divergence that exists in
the image benchmark.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .data import VideoCompressionResult


def aggregate_by_codec(results: list[VideoCompressionResult]) -> list[dict]:
    """Group by codec name; mean of every metric across all runs of that codec."""
    by_codec: dict[str, list[VideoCompressionResult]] = defaultdict(list)
    for r in results:
        by_codec[r.codec].append(r)

    summary: list[dict] = []
    for codec, rs in sorted(by_codec.items()):
        families = {r.codec_family for r in rs}
        summary.append({
            "codec": codec,
            "codec_family": sorted(families)[0] if families else "",
            "runs": len(rs),
            "psnr": float(np.mean([r.psnr for r in rs])),
            "ssim": float(np.mean([r.ssim for r in rs])),
            "bitrate_kbps": float(np.mean([r.bitrate_kbps for r in rs])),
            "bpp": float(np.mean([r.bpp for r in rs])),
            "ratio": float(np.mean([r.compression_ratio for r in rs])),
            "enc_fps": float(np.mean([r.enc_fps for r in rs])),
            "dec_fps": float(np.mean([r.dec_fps for r in rs])),
            "temporal": float(np.mean([r.temporal_metric for r in rs])),
            "size_kb": float(np.mean([r.compressed_bytes for r in rs])) / 1024,
        })
    return summary


def aggregate_by_codec_crf(results: list[VideoCompressionResult]) -> list[dict]:
    """Group by (codec, crf); mean of metrics across all sequences at that operating point.

    Used by the formal-test view: one row per (codec, crf) = 16 rows for 4 codecs × 4 CRFs,
    each averaged over all sequences in the (filtered) result set.
    """
    by_key: dict[tuple[str, int], list[VideoCompressionResult]] = defaultdict(list)
    for r in results:
        by_key[(r.codec, r.crf)].append(r)

    summary: list[dict] = []
    for (codec, crf), rs in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        families = {r.codec_family for r in rs}
        summary.append({
            "codec": codec,
            "codec_family": sorted(families)[0] if families else "",
            "crf": crf,
            "count": len(rs),
            "psnr": float(np.mean([r.psnr for r in rs])),
            "ssim": float(np.mean([r.ssim for r in rs])),
            "bitrate_kbps": float(np.mean([r.bitrate_kbps for r in rs])),
            "bpp": float(np.mean([r.bpp for r in rs])),
            "ratio": float(np.mean([r.compression_ratio for r in rs])),
            "enc_fps": float(np.mean([r.enc_fps for r in rs])),
            "dec_fps": float(np.mean([r.dec_fps for r in rs])),
            "temporal": float(np.mean([r.temporal_metric for r in rs])),
            "size_kb": float(np.mean([r.compressed_bytes for r in rs])) / 1024,
        })
    return summary


def aggregate_rd_curve(results: list[VideoCompressionResult]) -> dict[str, list[dict]]:
    """Per-codec RD curve: list of {crf, bpp, psnr, ssim, bitrate_kbps} sorted by bpp."""
    by_codec: dict[str, list[VideoCompressionResult]] = defaultdict(list)
    for r in results:
        by_codec[r.codec].append(r)

    curves: dict[str, list[dict]] = {}
    for codec, rs in by_codec.items():
        pts = sorted(
            [{"crf": r.crf, "bpp": r.bpp, "psnr": r.psnr,
              "ssim": r.ssim, "bitrate_kbps": r.bitrate_kbps}
             for r in rs],
            key=lambda p: p["bpp"],
        )
        curves[codec] = pts
    return curves


def bests(summary: Iterable[dict]) -> dict:
    """Pick best-in-class for the report's key-findings cards."""
    s = list(summary)
    if not s:
        return {}
    return {
        "best_psnr_codec": max(s, key=lambda r: r["psnr"])["codec"] if s else "",
        "best_psnr": max((r["psnr"] for r in s), default=0.0),
        "smallest_bpp_codec": min((r for r in s if r["bpp"] > 0),
                                   key=lambda r: r["bpp"], default=None),
        "fastest_enc_codec": max(s, key=lambda r: r["enc_fps"])["codec"] if s else "",
        "fastest_enc_fps": max((r["enc_fps"] for r in s), default=0.0),
        "best_ratio_codec": max(s, key=lambda r: r["ratio"])["codec"] if s else "",
    }
