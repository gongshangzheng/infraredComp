"""Generate HTML report from benchmark results."""

import os
import platform
from pathlib import Path
from datetime import datetime

from .metrics import CompressionResult


TEMPLATE_PATH = Path(__file__).parent / "report_template.html"


def _badge_class(codec: str) -> str:
    c = codec.lower()
    if "lossless" in c or codec == "PNG":
        return "badge-lossless"
    if "jpeg2000" in c or "j2k" in c:
        return "badge-j2k"
    if "avif" in c:
        return "badge-avif"
    if "webp" in c:
        return "badge-webp"
    if "jpeg" in c:
        return "badge-jpeg"
    if "png" in c:
        return "badge-png"
    # learned
    for k in ("bmshj", "mbt", "cheng", "hyperprior", "factorized", "anchor"):
        if k in c:
            return "badge-learned"
    return "badge-jpeg"


def _format_row(s: dict, best: dict) -> str:
    """Format one table row with optional highlighting."""
    psnr_cls = ' class="best-psnr"' if s["codec"] == best.get("psnr_codec") else ""
    bpp_cls = ' class="best-bpp"' if s["codec"] == best.get("bpp_codec") and s["bpp"] > 0 else ""
    enc_cls = ' class="best-speed"' if s["codec"] == best.get("enc_codec") else ""

    psnr_str = f'{s["psnr"]:.2f}' if s["psnr"] != float("inf") else '<span class="lossless-val">∞</span>'
    ssim_str = f'{s["ssim"]:.4f}' if s["ssim"] < 1.0 else '<span class="lossless-val">1.0000</span>'

    badge = _badge_class(s["codec"])

    return (
        f"<tr>"
        f'<td><span class="badge {badge}">{s["codec"]}</span></td>'
        f'<td>{s["n"]}</td>'
        f"<td{psnr_cls}>{psnr_str}</td>"
        f"<td>{ssim_str}</td>"
        f"<td{bpp_cls}>{s['bpp']:.2f}</td>"
        f'<td>{s["ratio"]:.1f}x</td>'
        f"<td{enc_cls}>{s['enc_ms']:.1f}</td>"
        f"<td>{s['dec_ms']:.1f}</td>"
        f"<td>{s['size_kb']:.1f}</td>"
        f"</tr>"
    )


def generate_html_report(
    results: list[CompressionResult],
    num_images: int,
    resolution: str = "640×512",
    bit_depth: str = "16-bit",
    output_dir: Path | None = None,
) -> str:
    """Generate HTML report from results. Returns path to generated file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by codec
    from collections import defaultdict
    by_codec: dict[str, list[CompressionResult]] = defaultdict(list)
    for r in results:
        by_codec[r.codec].append(r)

    summaries = []
    for codec, rs in sorted(by_codec.items()):
        import numpy as np
        summaries.append({
            "codec": codec,
            "n": len(rs),
            "psnr": float(np.mean([r.psnr for r in rs])),
            "ssim": float(np.mean([r.ssim for r in rs])),
            "bpp": float(np.mean([r.bpp for r in rs])),
            "ratio": float(np.mean([r.compression_ratio for r in rs])),
            "enc_ms": float(np.mean([r.encode_time_ms for r in rs])),
            "dec_ms": float(np.mean([r.decode_time_ms for r in rs])),
            "size_kb": float(np.mean([r.compressed_bytes for r in rs])) / 1024,
        })

    # Find bests (among lossy only for PSNR/BPP)
    lossy = [s for s in summaries if s["psnr"] != float("inf")]
    best = {
        "psnr_codec": max(lossy, key=lambda s: s["psnr"])["codec"] if lossy else "",
        "bpp_codec": min(lossy, key=lambda s: s["bpp"])["codec"] if lossy else "",
        "enc_codec": min(summaries, key=lambda s: s["enc_ms"])["codec"],
    }

    # Build table rows
    table_rows = "\n".join(_format_row(s, best) for s in summaries)

    # Key findings
    best_psnr = max(lossy, key=lambda s: s["psnr"]) if lossy else None
    best_ratio = max(summaries, key=lambda s: s["ratio"])
    best_enc = min(summaries, key=lambda s: s["enc_ms"])

    # Best RD: highest PSNR at lowest BPP among lossy
    # Use BD-rate proxy: pick the codec with best PSNR/BPP ratio improvement
    best_rd = None
    if lossy:
        # Simple heuristic: best PSNR at BPP < 1.0
        candidates = [s for s in lossy if s["bpp"] < 1.0]
        if candidates:
            best_rd = max(candidates, key=lambda s: s["psnr"])
        else:
            best_rd = max(lossy, key=lambda s: s["psnr"])

    # CPU info
    cpu = platform.processor() or platform.machine() or "Unknown"

    # Generate visual effect demo (image + HTML section)
    try:
        from .demo import demo_html_section, generate_demo_figure

        _, demo_image_name = generate_demo_figure(output_dir)
        demo_section = demo_html_section(demo_image_name)
    except Exception as e:
        demo_section = (
            f'\n  <!-- DEMO -->\n  <div class="section">\n'
            f'    <h2>Visual Effect Demo</h2>\n'
            f'    <p style="color: var(--text-dim);">Demo unavailable: {e}</p>\n'
            f"  </div>\n"
        )

    # Read template
    html = TEMPLATE_PATH.read_text()

    # Fill placeholders
    replacements = {
        "{{NUM_IMAGES}}": str(num_images),
        "{{RESOLUTION}}": resolution,
        "{{BIT_DEPTH}}": bit_depth,
        "{{DATE}}": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "{{CPU}}": cpu,
        "{{TABLE_ROWS}}": table_rows,
        "{{DEMO_SECTION}}": demo_section,
        "{{BEST_PSNR_VAL}}": f'{best_psnr["psnr"]:.1f} dB' if best_psnr else "N/A",
        "{{BEST_PSNR_CODEC}}": best_psnr["codec"] if best_psnr else "N/A",
        "{{BEST_RATIO_VAL}}": f'{best_ratio["ratio"]:.0f}x' if best_ratio else "N/A",
        "{{BEST_RATIO_CODEC}}": best_ratio["codec"] if best_ratio else "N/A",
        "{{BEST_ENC_VAL}}": f'{best_enc["enc_ms"]:.1f} ms' if best_enc else "N/A",
        "{{BEST_ENC_CODEC}}": best_enc["codec"] if best_enc else "N/A",
        "{{BEST_RD_VAL}}": f'{best_rd["psnr"]:.1f} dB @ {best_rd["bpp"]:.2f} bpp' if best_rd else "N/A",
        "{{BEST_RD_CODEC}}": best_rd["codec"] if best_rd else "N/A",
    }

    for k, v in replacements.items():
        html = html.replace(k, v)

    # Write output
    out_path = output_dir / "report.html"
    out_path.write_text(html)
    print(f"HTML report saved to: {out_path}")

    return str(out_path)
