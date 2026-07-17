"""Run all currently available contour-video codecs on Xiph CIF.

Collects every registered codec except dcvc_rt (which requires manual DCVC
checkpoint download) and sweeps each codec over its meaningful CRF/quality
levels.  Traditional codecs use x264-style CRFs; CompressAI image models use
their own quality levels; ssf2020 uses its 1-9 quality scale.

Results are merged into the existing results/video/xiph_cif.json so previously
completed runs are preserved and only missing ones are added.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.video import config
from benchmark.video.codecs import list_codecs
from benchmark.video.repro import build_metadata
from benchmark.video.stage1_extract import extract_contour_video
from benchmark.video.stage2_benchmark import benchmark_codec
from benchmark.video.data import VideoCompressionResult

DATASET_NAME = "Xiph-CIF-natural"
RESULTS_FILE_BY_MODE = {
    "formal": config.RESULTS_DIR / "xiph_cif.json",
    "speed": config.RESULTS_DIR / "xiph_cif_speed.json",
}

# CRF/quality sweep per codec family.  dcvc_rt is excluded because its
# CVPR-2025 checkpoints must be downloaded manually first.
CODEC_CRF_MAP: dict[str, list[int]] = {
    # Traditional ffmpeg codecs
    "x264": [18, 23, 28, 33],
    "x265": [18, 23, 28, 33],
    "vp9": [18, 23, 28, 33],
    "svtav1": [18, 23, 28, 33],
    # CompressAI learned video codec (quality 1-9, higher = more bits)
    "ssf2020": [1, 4, 7, 9],
    # CompressAI learned image codecs used per-frame as video codecs
    "img-bmshj2018-factorized": [1, 4, 8],
    "img-bmshj2018-hyperprior": [1, 4, 8],
    "img-mbt2018": [1, 4, 8],
    "img-mbt2018-mean": [1, 4, 8],
    "img-cheng2020-anchor": [1, 4, 6],
    "img-cheng2020-attn": [1, 4, 6],
    "img-ELIC": [1, 4, 5],
}

FRAME_CAP = None  # 默认不截断（speed/formal 全帧；想限帧显式传 --frames N）


def load_existing_results(path: Path) -> tuple[list[dict], dict]:
    """Load existing result JSON or return empty structures."""
    if not path.exists():
        return [], {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("runs", []), {k: v for k, v in data.items() if k != "runs"}


def result_key(run: dict) -> str:
    return f"{run.get('sequence_name')}|{run.get('codec')}|crf{run.get('crf')}"


def ensure_contour_artifacts(
    frames: int | None = FRAME_CAP,
    sequences: list[str] | None = None,
) -> list:
    """Extract/refresh canny contour artifacts for every Xiph CIF sequence."""
    xiph_dir = config.raw_dir("xiph_cif")
    seqs = sorted(xiph_dir.glob("*.y4m")) if xiph_dir.is_dir() else []
    if sequences:
        wanted = {s.strip() for s in sequences if s.strip()}
        seqs = [s for s in seqs if s.stem in wanted]
    artifacts = []
    for y4m in seqs:
        print(f"[stage1] {y4m.name} (frames={frames})")
        art = extract_contour_video(str(y4m), method="canny", frames=frames)
        print(f"[stage1] {y4m.name}: {art.frame_count} frames, {art.width}x{art.height}")
        artifacts.append(art)
    return artifacts


def _materialize_temp_frames(art):
    """If the artifact is video-based (PNGs deleted), decode the lossless
    contour video to a temp PNG dir (cropped to original dims) and return
    ``(temp_artifact, tmp_dir)``. Otherwise return ``(art, None)``.
    """
    if not art.video_path or art.frame_paths:
        return art, None
    import shutil
    import tempfile
    from dataclasses import replace
    from benchmark.video.ffmpeg_util import run_ffmpeg
    tmp_dir = Path(tempfile.mkdtemp(prefix="cvmany_"))
    run_ffmpeg(["-y", "-i", art.video_path, "-vsync", "0",
                "-vf", f"crop={art.width}:{art.height}:0:0",
                "-pix_fmt", "gray", str(tmp_dir / "frame_%06d.png")])
    frame_paths = [str(p) for p in sorted(tmp_dir.glob("frame_*.png"))]
    return replace(art, frames_dir=str(tmp_dir), frame_paths=frame_paths), tmp_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all available contour-video codecs on Xiph CIF.")
    ap.add_argument("--codecs", default=None,
                    help="comma-separated codec subset (default: all in CODEC_CRF_MAP)")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated seq stem subset (default: all Xiph CIF)")
    ap.add_argument("--frames", type=int, default=FRAME_CAP,
                    help=f"frame cap per sequence (default: {FRAME_CAP}=no cap)")
    ap.add_argument("--mode", default="formal", choices=["formal", "speed"],
                    help="formal→xiph_cif.json, speed→xiph_cif_speed.json(分文件)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing results; overwrite xiph_cif.json")
    args = ap.parse_args()

    config.ensure_dirs()
    RESULTS_FILE = RESULTS_FILE_BY_MODE[args.mode]
    existing_runs = [] if args.fresh else load_existing_results(RESULTS_FILE)[0]
    existing_keys = {result_key(r) for r in existing_runs}
    print(f"[init] {len(existing_runs)} existing runs, {len(existing_keys)} unique keys")

    run_order = [c for c in CODEC_CRF_MAP if c in list_codecs()]
    if args.codecs:
        wanted = {s.strip() for s in args.codecs.split(",") if s.strip()}
        run_order = [c for c in run_order if c in wanted]

    artifacts = ensure_contour_artifacts(
        args.frames,
        sequences=args.sequences.split(",") if args.sequences else None,
    )
    if not artifacts:
        print("error: no contour artifacts", file=sys.stderr)
        return 1

    all_runs = list(existing_runs)
    codecs_used = []
    crfs_used: list[int] = []

    def _flush_save():
        """Incremental save so a crash mid-run doesn't lose completed runs."""
        payload = {
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "runs": all_runs,
            "mode": args.mode,
        }
        meta = build_metadata(
            inputs=[str(a.manifest_path) for a in artifacts],
            codecs=codecs_used or run_order,
            crfs=sorted(crfs_used) if crfs_used else sorted({c for crfs in CODEC_CRF_MAP.values() for c in crfs}),
            method="canny", frame_cap=args.frames,
            runner="scripts/run_all_video_models.py", dataset=DATASET_NAME,
        )
        payload.update(meta)
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for art in artifacts:
        # benchmark_codec reads artifact.frame_paths (PNGs deleted in the new
        # pipeline). Materialize transient PNG frames once per sequence and
        # reuse across the codec grid.
        work_art, tmp_dir = _materialize_temp_frames(art)
        try:
            for codec_name in run_order:
                if codec_name not in CODEC_CRF_MAP:
                    print(f"  SKIP {codec_name}: no CRF mapping defined")
                    continue
                crfs = CODEC_CRF_MAP[codec_name]
                for crf in crfs:
                    key = f"{art.source_name}|{codec_name}|crf{crf}"
                    if key in existing_keys:
                        print(f"  SKIP {key}: already exists")
                        continue
                    try:
                        r = benchmark_codec(work_art, codec_name, crf, dataset=DATASET_NAME)
                        all_runs.append(r.to_dict())
                        existing_keys.add(key)
                        if codec_name not in codecs_used:
                            codecs_used.append(codec_name)
                        if crf not in crfs_used:
                            crfs_used.append(crf)
                        print(f"  OK   {key} PSNR={r.psnr:.2f} SSIM={r.ssim:.4f}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        # Truncate verbose ffmpeg stderr to the first meaningful line.
                        msg = str(e).strip().splitlines()
                        short = msg[0] if msg else str(e)
                        print(f"  ERROR {key}: {short[:160]}", flush=True)
            _flush_save()  # persist after each sequence
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    _flush_save()
    print(f"[done] {len(all_runs)} runs -> {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
