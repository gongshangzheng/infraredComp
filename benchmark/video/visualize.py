"""Visualization + CSV/Markdown for the video benchmark.

Mirrors the image benchmark's plot pattern (fig/ax, savefig dpi=150, close).
Charts are written to results/video/charts/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tabulate import tabulate  # noqa: E402

from . import config  # noqa: E402
from .aggregate import aggregate_by_codec, aggregate_rd_curve, bests  # noqa: E402
from .data import VideoCompressionResult  # noqa: E402

OUTPUT_DIR = config.CHARTS_DIR

_VIDEO_CODEC_STYLES = {
    "x264": {"color": "#ef4444", "marker": "o"},
    "x265": {"color": "#3b82f6", "marker": "s"},
    "svtav1": {"color": "#10b981", "marker": "^"},
    "vp9": {"color": "#f59e0b", "marker": "D"},
}


def _style(codec: str) -> dict:
    return _VIDEO_CODEC_STYLES.get(codec, {"color": "#6b7280", "marker": "o"})


def _ensure() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _save(fig, name: str) -> Path:
    path = _ensure() / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  chart -> {path}")
    return path


def plot_rd_psnr_vs_bitrate(curves: dict, path: Path | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for codec, pts in curves.items():
        st = _style(codec)
        xs = [p["bitrate_kbps"] for p in pts]
        ys = [p["psnr"] for p in pts]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], st["marker"] + "-",
                color=st["color"], label=codec, markersize=6, linewidth=1.5)
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("RD Curve: PSNR vs Bitrate")
    ax.legend(); ax.grid(True, alpha=0.3)
    return _save(fig, path.name if path else "rd_psnr_vs_bitrate.png")


def plot_rd_ssim_vs_bitrate(curves: dict, path: Path | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for codec, pts in curves.items():
        st = _style(codec)
        xs = [p["bitrate_kbps"] for p in pts]
        ys = [p["ssim"] for p in pts]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], st["marker"] + "-",
                color=st["color"], label=codec, markersize=6, linewidth=1.5)
    ax.set_xlabel("Bitrate (kbps)"); ax.set_ylabel("SSIM")
    ax.set_title("RD Curve: SSIM vs Bitrate"); ax.legend(); ax.grid(True, alpha=0.3)
    return _save(fig, path.name if path else "rd_ssim_vs_bitrate.png")


def _bar(summary: list[dict], key: str, ylabel: str, title: str, fname: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    codecs = [s["codec"] for s in summary]
    vals = [s[key] for s in summary]
    colors = [_style(c)["color"] for c in codecs]
    ax.bar(codecs, vals, color=colors)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, fname)


def plot_encode_fps(summary, path=None): return _bar(summary, "enc_fps", "fps", "Encode FPS", "encode_fps.png")
def plot_decode_fps(summary, path=None): return _bar(summary, "dec_fps", "fps", "Decode FPS", "decode_fps.png")
def plot_temporal(summary, path=None): return _bar(summary, "temporal", "std(PSNR)", "Temporal Consistency (lower=better)", "temporal_consistency.png")
def plot_ratio(summary, path=None): return _bar(summary, "ratio", "ratio", "Compression Ratio", "compression_ratio.png")


def save_csv(summary: list[dict], path: Path) -> Path:
    headers = ["codec", "codec_family", "runs", "psnr", "ssim", "bitrate_kbps",
               "bpp", "ratio", "enc_fps", "dec_fps", "temporal", "size_kb"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for s in summary:
            f.write(",".join(
                f'{s[h]:.4f}' if isinstance(s.get(h), float) else str(s.get(h, ""))
                for h in headers
            ) + "\n")
    print(f"  csv -> {path}")
    return path


def save_markdown_table(summary: list[dict], path: Path) -> Path:
    headers = ["Codec", "Runs", "PSNR(dB)", "SSIM", "BPP", "Ratio", "Enc fps", "Dec fps", "Temporal"]
    rows = []
    for s in summary:
        rows.append([
            s["codec"], s["runs"], f'{s["psnr"]:.2f}', f'{s["ssim"]:.4f}',
            f'{s["bpp"]:.3f}', f'{s["ratio"]:.1f}x',
            f'{s["enc_fps"]:.1f}', f'{s["dec_fps"]:.1f}', f'{s["temporal"]:.2f}',
        ])
    md = tabulate(rows, headers=headers, tablefmt="pipe")
    path.write_text("# Contour Video Compression Results\n\n" + md + "\n", encoding="utf-8")
    print(f"  md -> {path}")
    return path


def print_console_table(summary: list[dict]) -> None:
    headers = ["Codec", "Runs", "PSNR", "SSIM", "BPP", "Ratio", "Enc fps", "Dec fps"]
    rows = [[s["codec"], s["runs"], f'{s["psnr"]:.2f}', f'{s["ssim"]:.4f}',
             f'{s["bpp"]:.3f}', f'{s["ratio"]:.1f}x',
             f'{s["enc_fps"]:.1f}', f'{s["dec_fps"]:.1f}'] for s in summary]
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def generate_report(results: list[VideoCompressionResult]) -> dict:
    """Top-level: aggregate → console + csv + md + all charts. Returns summary."""
    summary = aggregate_by_codec(results)
    curves = aggregate_rd_curve(results)

    print("\n=== Per-codec summary ===")
    print_console_table(summary)

    save_csv(summary, OUTPUT_DIR.parent / "results.csv")
    save_markdown_table(summary, OUTPUT_DIR.parent / "results.md")

    if curves:
        plot_rd_psnr_vs_bitrate(curves)
        plot_rd_ssim_vs_bitrate(curves)
    if summary:
        plot_encode_fps(summary)
        plot_decode_fps(summary)
        plot_temporal(summary)
        plot_ratio(summary)

    return {"summary": summary, "bests": bests(summary)}
