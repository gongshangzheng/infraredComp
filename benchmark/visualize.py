"""Image compression benchmark — visualization and reporting."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from .metrics import CompressionResult

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def aggregate_by_codec(results: list[CompressionResult]) -> list[dict]:
    """Aggregate results per codec (mean across images)."""
    from collections import defaultdict

    by_codec: dict[str, list[CompressionResult]] = defaultdict(list)
    for r in results:
        by_codec[r.codec].append(r)

    summary = []
    for codec, rs in sorted(by_codec.items()):
        n = len(rs)
        summary.append({
            "Codec": codec,
            "Images": n,
            "PSNR": float(np.mean([r.psnr for r in rs])),
            "SSIM": float(np.mean([r.ssim for r in rs])),
            "BPP": float(np.mean([r.bpp for r in rs])),
            "Ratio": float(np.mean([r.compression_ratio for r in rs])),
            "Enc_ms": float(np.mean([r.encode_time_ms for r in rs])),
            "Dec_ms": float(np.mean([r.decode_time_ms for r in rs])),
            "Size_KB": float(np.mean([r.compressed_bytes for r in rs])) / 1024,
        })
    return summary


def save_csv(summary: list[dict], path: Path):
    """Save summary table as CSV."""
    headers = ["Codec", "Images", "PSNR", "SSIM", "BPP", "Ratio", "Enc_ms", "Dec_ms", "Size_KB"]
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in summary:
            vals = []
            for h in headers:
                v = row.get(h, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    print(f"CSV saved to: {path}")


def save_markdown_table(summary: list[dict], path: Path):
    """Save summary table as Markdown."""
    headers = ["Codec", "Images", "PSNR (dB)", "SSIM", "BPP", "Ratio", "Enc ms", "Dec ms", "Size KB"]
    rows = []
    for s in summary:
        rows.append([
            s["Codec"],
            str(s["Images"]),
            f'{s["PSNR"]:.2f}',
            f'{s["SSIM"]:.4f}',
            f'{s["BPP"]:.2f}',
            f'{s["Ratio"]:.1f}x',
            f'{s["Enc_ms"]:.1f}',
            f'{s["Dec_ms"]:.1f}',
            f'{s["Size_KB"]:.1f}',
        ])
    md = tabulate(rows, headers=headers, tablefmt="pipe")
    with open(path, "w") as f:
        f.write("# Compression Benchmark Results\n\n")
        f.write(md + "\n")
    print(f"Markdown table saved to: {path}")


def print_console_table(summary: list[dict]):
    """Print results to console."""
    headers = ["Codec", "Images", "PSNR (dB)", "SSIM", "BPP", "Ratio", "Enc ms", "Dec ms", "Size KB"]
    rows = []
    for s in summary:
        rows.append([
            s["Codec"],
            str(s["Images"]),
            f'{s["PSNR"]:.2f}',
            f'{s["SSIM"]:.4f}',
            f'{s["BPP"]:.2f}',
            f'{s["Ratio"]:.1f}x',
            f'{s["Enc_ms"]:.1f}',
            f'{s["Dec_ms"]:.1f}',
            f'{s["Size_KB"]:.1f}',
        ])
    print("\n" + "=" * 100)
    print("COMPRESSION BENCHMARK RESULTS")
    print("=" * 100)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Per-codec visual style: unique color + marker + hatch pattern.
# This makes it easy to distinguish compression methods on plots that use
# lines/scatter points as well as on bar charts.
_CODEC_STYLES = {
    "AVIF": {"color": "#2196F3", "marker": "o", "hatch": ""},       # blue circle
    "AVIF-lossless": {"color": "#2196F3", "marker": "s", "hatch": ""}, # blue square
    "JPEG": {"color": "#F44336", "marker": "D", "hatch": ""},       # red diamond
    "JPEG2000": {"color": "#FF9800", "marker": "^", "hatch": "/"},  # orange triangle
    "JPEG2000-lossless": {"color": "#FF9800", "marker": "v", "hatch": "/"},
    "PNG": {"color": "#4CAF50", "marker": "p", "hatch": "\\"},     # green pentagon
    "WebP": {"color": "#9C27B0", "marker": "h", "hatch": "x"},      # purple hexagon
    "WebP-lossless": {"color": "#9C27B0", "marker": "8", "hatch": "x"},
    # Learned / neural codecs
    "bmshj": {"color": "#00BCD4", "marker": "P", "hatch": "+"},    # cyan filled plus
    "mbt": {"color": "#795548", "marker": "X", "hatch": "-"},       # brown cross
    "cheng": {"color": "#FFEB3B", "marker": "*", "hatch": "|"},     # yellow star
}


def _codec_family(codec: str) -> str:
    """Return canonical family name used for styling."""
    c = codec.strip()
    # Exact matches first (handles lossless variants).
    if c in _CODEC_STYLES:
        return c
    # Learned codec families.
    lower = c.lower()
    for family in ("bmshj", "mbt", "cheng"):
        if family in lower:
            return family
    # Traditional families (order matters: JPEG2000 before JPEG).
    for family in ("JPEG2000", "WebP", "JPEG", "AVIF"):
        if c.startswith(family):
            return family
    if c == "PNG":
        return "PNG"
    return c


def _codec_style(codec: str) -> dict:
    """Return the visual style dictionary for a codec."""
    family = _codec_family(codec)
    return _CODEC_STYLES.get(family, {"color": "#607D8B", "marker": "o", "hatch": ""})


def plot_psnr_vs_bpp(summary: list[dict], path: Path):
    """RD curve: PSNR vs BPP."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate lossless (PSNR=inf) — plot at the top
    finite = [s for s in summary if np.isfinite(s["PSNR"])]
    lossless = [s for s in summary if not np.isfinite(s["PSNR"])]

    # Group finite points by codec family for connecting lines
    groups = {}
    for s in finite:
        base = s["Codec"].split(" q")[0].split("-q")[0]
        groups.setdefault(base, []).append(s)

    for base, points in sorted(groups.items()):
        points.sort(key=lambda x: x["BPP"])
        bpp = [p["BPP"] for p in points]
        psnr = [p["PSNR"] for p in points]
        style = _codec_style(base)
        ax.plot(bpp, psnr, marker=style["marker"], color=style["color"],
                label=base, linewidth=1.5, markersize=8)

    # Annotate lossless codecs at the top
    if lossless:
        top_psnr = max(p["PSNR"] for p in finite) if finite else 50
        for i, s in enumerate(lossless):
            ax.annotate(
                f'{s["Codec"]} (lossless)',
                xy=(s["BPP"], top_psnr + 2 + i * 1.5),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                xytext=(s["BPP"], top_psnr + 4 + i * 2),
            )

    ax.set_xlabel("Bits Per Pixel (BPP)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion: PSNR vs BPP", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_ssim_vs_bpp(summary: list[dict], path: Path):
    """RD curve: SSIM vs BPP."""
    fig, ax = plt.subplots(figsize=(10, 7))

    finite = [s for s in summary if s["SSIM"] < 1.0]

    groups = {}
    for s in finite:
        base = s["Codec"].split(" q")[0].split("-q")[0]
        groups.setdefault(base, []).append(s)

    for base, points in sorted(groups.items()):
        points.sort(key=lambda x: x["BPP"])
        bpp = [p["BPP"] for p in points]
        ssim = [p["SSIM"] for p in points]
        style = _codec_style(base)
        ax.plot(bpp, ssim, marker=style["marker"], color=style["color"],
                label=base, linewidth=1.5, markersize=8)

    ax.set_xlabel("Bits Per Pixel (BPP)", fontsize=12)
    ax.set_ylabel("SSIM", fontsize=12)
    ax.set_title("Rate-Distortion: SSIM vs BPP", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_psnr_vs_encoding_time(summary: list[dict], path: Path):
    """Quality vs speed tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 7))

    finite = [s for s in summary if np.isfinite(s["PSNR"])]

    # Group by codec family so each method gets a single legend entry with a
    # distinct color and marker shape.
    groups: dict[str, list[dict]] = {}
    for s in finite:
        family = _codec_family(s["Codec"])
        groups.setdefault(family, []).append(s)

    for family, points in sorted(groups.items()):
        style = _codec_style(family)
        x = [p["Enc_ms"] for p in points]
        y = [p["PSNR"] for p in points]
        ax.scatter(x, y, marker=style["marker"], color=style["color"],
                   s=100, zorder=5, label=family)
        for p in points:
            ax.annotate(
                p["Codec"],
                xy=(p["Enc_ms"], p["PSNR"]),
                fontsize=7, textcoords="offset points",
                xytext=(5, 5),
            )

    ax.set_xlabel("Encoding Time (ms)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Quality vs Encoding Speed", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_compression_ratio_bar(summary: list[dict], path: Path):
    """Bar chart of compression ratios."""
    fig, ax = plt.subplots(figsize=(12, 6))

    codecs = [s["Codec"] for s in summary]
    ratios = [s["Ratio"] for s in summary]
    styles = [_codec_style(c) for c in codecs]
    colors = [s["color"] for s in styles]
    hatches = [s["hatch"] for s in styles]

    bars = ax.barh(codecs, ratios, color=colors, edgecolor="white", linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_title("Compression Ratio by Codec", fontsize=14)
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="No compression")

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        label = f"{ratio:.1f}x"
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8)

    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_psnr_bar(summary: list[dict], path: Path):
    """Bar chart of PSNR values (lossless excluded)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    finite = [s for s in summary if np.isfinite(s["PSNR"])]
    codecs = [s["Codec"] for s in finite]
    psnrs = [s["PSNR"] for s in finite]
    styles = [_codec_style(c) for c in codecs]
    colors = [s["color"] for s in styles]
    hatches = [s["hatch"] for s in styles]

    bars = ax.barh(codecs, psnrs, color=colors, edgecolor="white", linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.set_xlabel("PSNR (dB)", fontsize=12)
    ax.set_title("PSNR by Codec (higher is better)", fontsize=14)

    for bar, v in zip(bars, psnrs):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=8)

    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_bpp_bar(summary: list[dict], path: Path):
    """Bar chart of BPP values."""
    fig, ax = plt.subplots(figsize=(12, 6))

    codecs = [s["Codec"] for s in summary]
    bpp = [s["BPP"] for s in summary]
    styles = [_codec_style(c) for c in codecs]
    colors = [s["color"] for s in styles]
    hatches = [s["hatch"] for s in styles]

    bars = ax.barh(codecs, bpp, color=colors, edgecolor="white", linewidth=0.5)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax.set_xlabel("Bits Per Pixel (BPP)", fontsize=12)
    ax.set_title("BPP by Codec (lower is better)", fontsize=14)
    ax.axvline(x=8, color="red", linestyle="--", alpha=0.5, label="Uncompressed (8-bit)")

    for bar, v in zip(bars, bpp):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)

    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_speed_bar(summary: list[dict], path: Path):
    """Bar chart comparing encode and decode times."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    codecs = [s["Codec"] for s in summary]
    enc_times = [s["Enc_ms"] for s in summary]
    dec_times = [s["Dec_ms"] for s in summary]
    styles = [_codec_style(c) for c in codecs]
    colors = [s["color"] for s in styles]
    hatches = [s["hatch"] for s in styles]

    bars1 = ax1.barh(codecs, enc_times, color=colors, edgecolor="white", linewidth=0.5)
    for bar, hatch in zip(bars1, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax1.set_xlabel("Time (ms)", fontsize=11)
    ax1.set_title("Encoding Speed", fontsize=13)
    ax1.set_xscale("log")
    ax1.grid(True, axis="x", alpha=0.3)

    bars2 = ax2.barh(codecs, dec_times, color=colors, edgecolor="white", linewidth=0.5)
    for bar, hatch in zip(bars2, hatches):
        if hatch:
            bar.set_hatch(hatch)
    ax2.set_xlabel("Time (ms)", fontsize=11)
    ax2.set_title("Decoding Speed", fontsize=13)
    ax2.set_xscale("log")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def generate_all_plots(summary: list[dict]):
    """Generate all visualization plots."""
    out_dir = ensure_output_dir()

    plot_psnr_vs_bpp(summary, out_dir / "rd_curve_psnr_bpp.png")
    plot_ssim_vs_bpp(summary, out_dir / "rd_curve_ssim_bpp.png")
    plot_psnr_vs_encoding_time(summary, out_dir / "quality_vs_speed.png")
    plot_compression_ratio_bar(summary, out_dir / "compression_ratio.png")
    plot_psnr_bar(summary, out_dir / "psnr_comparison.png")
    plot_bpp_bar(summary, out_dir / "bpp_comparison.png")
    plot_speed_bar(summary, out_dir / "speed_comparison.png")


def generate_report(results: list[CompressionResult]):
    """Generate full report: console table, CSV, markdown, and plots."""
    summary = aggregate_by_codec(results)
    out_dir = ensure_output_dir()

    print_console_table(summary)
    save_csv(summary, out_dir / "results.csv")
    save_markdown_table(summary, out_dir / "results.md")
    generate_all_plots(summary)

    print(f"\nAll results saved to: {out_dir}/")
