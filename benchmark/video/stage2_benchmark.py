"""Stage 2 — benchmark standard video codecs on a contour video.

For each (codec, CRF) pair: encode the lossless contour PNG sequence to a
bitstream, decode back to PNGs, then measure per-frame PSNR/SSIM vs the
ground-truth contour frames, plus bitrate, fps, and temporal consistency.

Reads/writes ``results/video/results.json`` (the artifact the FastAPI backend
serves). Execution is decoupled from reporting.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from benchmark.metrics import timed
from . import config
from .codecs import build_codec, list_codecs
from .data import VideoCompressionResult
from .data import ContourArtifact
from .ffmpeg_util import get_duration_seconds
from .metrics import (
    fps_from_timed,
    mean_psnr,
    mean_ssim,
    per_frame_quality,
    temporal_consistency,
)
from .stage1_extract import load_contour_frames


def _read_recon_frames(recon_dir: Path) -> np.ndarray:
    """Load decoded PNGs (sorted) into an (N, H, W) uint8 stack."""
    files = sorted(recon_dir.glob("frame_*.png"))
    if not files:
        raise RuntimeError(f"No reconstructed frames in {recon_dir}")
    frames = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    return np.stack(frames, axis=0)


def synthesize_recon_video(recon_dir: Path, fps: float, mp4_path: str) -> str:
    """Encode the decoded recon PNG sequence into a viewable mp4.

    Neural codecs (ssf2020 / img-* / dcvc_rt) emit a ``.bin`` bitstream that no
    browser can play, but their decoded recon frames are written as PNGs. This
    lossless-ish x264 pass turns those PNGs into a playable mp4 cached next to
    the bitstream (``bitstreams/{tag}.mp4``), so the backend's ``_bitstream_for``
    finds it and the speed/formal pages can show the reconstruction. Best-effort:
    callers log failures rather than failing the whole run.
    """
    from .ffmpeg_util import run_ffmpeg
    Path(mp4_path).parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-y",
        "-framerate", str(fps if fps and fps > 0 else 25.0),
        "-i", str(Path(recon_dir) / "frame_%06d.png"),
        # pad odd dims to even (yuv420p requirement); no-op for even dims
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        mp4_path,
    ]
    run_ffmpeg(args)
    return mp4_path


def benchmark_codec(
    artifact: ContourArtifact,
    codec_name: str,
    crf: int,
    preset: str | None = None,
    dataset: str = "",
) -> VideoCompressionResult:
    """Run one codec @ one CRF on the contour video; return a result row."""
    config.ensure_dirs()
    codec = build_codec(codec_name, crf=crf, preset=preset)

    frames_dir = artifact.frames_dir
    # tag 含 method: canny/sobel/hed 同 seq×codec×crf 的输出文件互不覆盖。
    method = artifact.method or "canny"
    tag = f"{artifact.source_name}_{method}_{codec_name}_crf{crf}"
    bitstream = str(Path(config.BITSTREAMS_DIR) / f"{tag}.{codec.ext}")
    recon_dir = Path(config.RECON_DIR) / tag
    if recon_dir.exists():
        shutil.rmtree(recon_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)

    if getattr(codec, "is_neural", False):
        # ---- neural in-process path (learned video codecs: ssf2020 / dcvc-rt) ----
        import cv2
        gt_frames = load_contour_frames(artifact)  # np.ndarray (N,H,W) uint8
        frame_list = [gt_frames[i] for i in range(len(gt_frames))]

        def _encode():
            bs = codec.encode_inprocess(frame_list, artifact.fps)
            Path(bitstream).write_bytes(bs)
            return bs
        _, encode_ms = timed(_encode)

        # estimate 模式(learned_image):用 codec 报的估计字节数,非文件大小(bitstream 仅元数据)。
        compressed_bytes = getattr(codec, '_estimated_bytes', None) or (
            os.path.getsize(bitstream) if os.path.exists(bitstream) else 0)
        duration_s = (
            artifact.frame_count / artifact.fps
            if artifact.fps > 0 and artifact.frame_count > 0
            else 1.0
        )
        bitrate_kbps = (compressed_bytes * 8 / duration_s) if duration_s > 0 else 0.0

        def _decode():
            bs = Path(bitstream).read_bytes()
            rec_frames = codec.decode_inprocess(bs, len(frame_list), (artifact.height, artifact.width))
            # write recon frames as PNGs so the shared metrics pipeline works
            for i, fr in enumerate(rec_frames):
                if fr.ndim == 3 and fr.shape[2] == 1:
                    fr = fr[:, :, 0]
                cv2.imwrite(str(recon_dir / f"frame_{i:06d}.png"), fr)
        _, decode_ms = timed(_decode)

        # Neural bitstreams are .bin (not browser-playable): synthesize a
        # viewable mp4 from the decoded recon PNGs so the pages can show it.
        mp4_path = str(Path(config.BITSTREAMS_DIR) / f"{tag}.mp4")
        try:
            synthesize_recon_video(recon_dir, artifact.fps, mp4_path)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: recon-video synth failed for {tag}: {e}")
    else:
        # ---- ffmpeg path (legacy: x264/x265/svtav1/vp9) ----
        # 1. Encode (timed wall-clock around the ffmpeg subprocess)
        def _encode():
            from .ffmpeg_util import run_ffmpeg
            run_ffmpeg(codec.encode_args(frames_dir, artifact.fps, bitstream))

        _, encode_ms = timed(_encode)

        compressed_bytes = os.path.getsize(bitstream) if os.path.exists(bitstream) else 0
        duration_s = (
            artifact.frame_count / artifact.fps
            if artifact.fps > 0 and artifact.frame_count > 0
            else get_duration_seconds(bitstream)
        )
        bitrate_kbps = (compressed_bytes * 8 / duration_s) if duration_s > 0 else 0.0

        # 2. Decode (timed)
        def _decode():
            from .ffmpeg_util import run_ffmpeg
            run_ffmpeg(codec.decode_args(bitstream, str(recon_dir)))

        _, decode_ms = timed(_decode)

    # 3. Load GT + reconstructed, align by index
    gt = load_contour_frames(artifact)
    rec = _read_recon_frames(recon_dir)

    # Crop reconstructed (possibly padded to even) back to artifact WxH.
    h, w = artifact.height, artifact.width
    if rec.shape[1] != h or rec.shape[2] != w:
        rec = rec[:, :h, :w]
    n = min(len(gt), len(rec))

    per_psnr, per_ssim = per_frame_quality(gt[:n], rec[:n])
    psnr = mean_psnr(per_psnr)
    ssim = mean_ssim(per_ssim)
    temporal = temporal_consistency(per_psnr)

    pixel_count = n * h * w
    bpp = (compressed_bytes * 8 / pixel_count) if pixel_count else 0.0
    original_bytes = int(gt.nbytes) or (n * h * w)
    ratio = (original_bytes / compressed_bytes) if compressed_bytes else 0.0

    enc_fps = fps_from_timed(n, encode_ms)
    dec_fps = fps_from_timed(n, decode_ms)

    decoded_sample = ""
    samples = sorted(recon_dir.glob("frame_*.png"))
    if samples:
        decoded_sample = str(samples[0])

    return VideoCompressionResult(
        id=f"{artifact.source_name}|{method}|{codec_name}|crf{crf}",
        codec=codec_name,
        codec_family=codec.family,
        crf=crf,
        sequence_name=artifact.source_name,
        method=artifact.method,
        frame_count=n,
        fps=artifact.fps,
        width=w,
        height=h,
        psnr=psnr,
        ssim=ssim,
        per_frame_psnr=per_psnr,
        per_frame_ssim=per_ssim,
        bitrate_kbps=bitrate_kbps,
        bpp=bpp,
        compression_ratio=ratio,
        compressed_bytes=compressed_bytes,
        duration_s=duration_s,
        encode_time_ms=encode_ms,
        decode_time_ms=decode_ms,
        enc_fps=enc_fps,
        dec_fps=dec_fps,
        temporal_metric=temporal,
        decoded_sample=decoded_sample,
        dataset=dataset,
    )


def run_benchmark(
    artifact: ContourArtifact,
    codecs: list[str] | None = None,
    crfs: list[int] | None = None,
    preset: str | None = None,
    save: bool = True,
    dataset: str = "",
) -> list[VideoCompressionResult]:
    """Run a (codecs × crfs) grid on one contour video.

    With save=True (default) persist results.json afterwards. Callers that
    accumulate across multiple sequences pass save=False and write once via
    save_results_json().
    """
    if codecs is None:
        codecs = list_codecs()
    if crfs is None:
        crfs = [18, 23, 28, 33]

    # Video-based artifact (PNGs deleted): materialize transient frames ONCE
    # from the lossless contour video — cropped back to the original (possibly
    # odd) dims — and reuse across the whole (codecs x crfs) grid. The temp dir
    # is cleaned up after the run; the persistent contour dir keeps no PNGs.
    work_artifact = artifact
    tmp_frames_dir: Path | None = None
    if artifact.video_path and not artifact.frame_paths:
        import tempfile
        from .ffmpeg_util import run_ffmpeg as _rf
        tmp_frames_dir = Path(tempfile.mkdtemp(prefix="cvframes_"))
        _rf(["-y", "-i", artifact.video_path, "-vsync", "0",
             "-vf", f"crop={artifact.width}:{artifact.height}:0:0",
             "-pix_fmt", "gray", str(tmp_frames_dir / "frame_%06d.png")])
        frame_paths = [str(p) for p in sorted(tmp_frames_dir.glob("frame_*.png"))]
        work_artifact = _replace_artifact(artifact, frames_dir=str(tmp_frames_dir),
                                          frame_paths=frame_paths)

    results: list[VideoCompressionResult] = []
    total = len(codecs) * len(crfs)
    done = 0
    try:
        for codec_name in codecs:
            for crf in crfs:
                try:
                    r = benchmark_codec(work_artifact, codec_name, crf, preset, dataset)
                    results.append(r)
                except Exception as e:  # noqa: BLE001
                    print(f"  ERROR [{codec_name} crf{crf}] on {artifact.source_name}: {e}")
                done += 1
                print(f"  [{codec_name} crf{crf}] done ({done}/{total})")
    finally:
        if tmp_frames_dir:
            shutil.rmtree(tmp_frames_dir, ignore_errors=True)

    if save:
        save_results_json(results)
    return results


def _replace_artifact(artifact: ContourArtifact, **changes) -> ContourArtifact:
    """dataclasses.replace for ContourArtifact (keep all fields, override a few)."""
    from dataclasses import replace
    return replace(artifact, **changes)


def save_results_json(
    results: list[VideoCompressionResult],
    path: str | Path | None = None,
    metadata: dict | None = None,
) -> Path:
    """Persist results to results/video/results.json (overwrite).

    Optional metadata (codec/crf/git-sha/dataset envelope) is merged at the top
    level alongside generated_at/runs; the server tolerates the extra keys.
    """
    out = Path(path) if path else config.RESULTS_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "runs": [r.to_dict() for r in results],
    }
    if metadata:
        payload.update(metadata)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} results to {out}")
    return out
