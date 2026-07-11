"""Generate a standalone HTML report from video benchmark results."""

from __future__ import annotations

from pathlib import Path

from . import config
from .aggregate import aggregate_by_codec, bests
from .data import VideoCompressionResult

TEMPLATE_PATH = Path(__file__).parent / "video_report_template.html"


def _cards(summary: list[dict], b: dict) -> str:
    if not summary:
        return ""
    items = [
        ("最佳 PSNR", f'{b.get("best_psnr", 0):.2f} dB', b.get("best_psnr_codec", "")),
        ("最快编码", f'{b.get("fastest_enc_fps", 0):.1f} fps', b.get("fastest_enc_codec", "")),
        ("最高压缩比", next((f'{s["ratio"]:.1f}x' for s in summary if s["codec"] == b.get("best_ratio_codec")), "-"), b.get("best_ratio_codec", "")),
    ]
    return "".join(
        f'<div class="card"><div class="label">{label}</div><div class="value">{val}</div><div class="label">{name}</div></div>'
        for label, val, name in items
    )


def _table(summary: list[dict], b: dict) -> str:
    if not summary:
        return ""
    head = "<tr><th>Codec</th><th>Runs</th><th>PSNR</th><th>SSIM</th><th>BPP</th><th>Ratio</th><th>Enc fps</th><th>Dec fps</th></tr>"
    rows = []
    for s in summary:
        is_best_psnr = s["codec"] == b.get("best_psnr_codec")
        psnr_cls = ' class="best"' if is_best_psnr else ""
        rows.append(
            f"<tr><td>{s['codec']}</td><td>{s['runs']}</td>"
            f"<td{psnr_cls}>{s['psnr']:.2f}</td><td>{s['ssim']:.4f}</td>"
            f"<td>{s['bpp']:.3f}</td><td>{s['ratio']:.1f}x</td>"
            f"<td>{s['enc_fps']:.1f}</td><td>{s['dec_fps']:.1f}</td></tr>"
        )
    return head + "".join(rows)


def _charts() -> str:
    names = [
        "rd_psnr_vs_bitrate.png", "rd_ssim_vs_bitrate.png",
        "encode_fps.png", "decode_fps.png",
        "temporal_consistency.png", "compression_ratio.png",
    ]
    cards = []
    for n in names:
        p = config.CHARTS_DIR / n
        if p.exists():
            cards.append(f'<div class="chart-card"><img src="{p}" alt="{n}"/></div>')
    return "".join(cards)


def generate_html_report(results: list[VideoCompressionResult], output_dir: Path | None = None) -> str:
    summary = aggregate_by_codec(results)
    b = bests(summary)
    tpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    meta = f"{len(results)} runs across {len(summary)} codecs"
    html = (tpl
            .replace("{{META}}", meta)
            .replace("{{CARDS}}", _cards(summary, b))
            .replace("{{TABLE}}", _table(summary, b))
            .replace("{{CHARTS}}", _charts()))
    out = (output_dir or config.RESULTS_DIR) / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  html -> {out}")
    return str(out)
