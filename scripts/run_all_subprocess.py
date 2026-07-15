"""Orchestrate the full codec sweep with per-(sequence, codec) subprocess isolation.

A native segfault in compressai's rans entropy coder accumulates across many
in-process runs and kills the whole Python process (SIGSEGV, exit 139) — taking
all buffered/unflushed results with it.  To survive it, each (sequence, codec)
pair runs in its OWN subprocess (``scripts/bench_one.py``) which handles all
CRFs for that codec.  The parent reads RESULT:/ERROR: lines from stdout, so any
results emitted before a crash are kept; a crashed subprocess just means the
remaining CRFs of that one codec on that one sequence are lost (and re-runnable).

Stage 1 (contour extraction) is done once per sequence up front so the 6
contour dirs exist; bench_one reuses them via --contour-dir.

Usage:
    python -u scripts/run_all_subprocess.py --frames 10
    python -u scripts/run_all_subprocess.py --frames 10 --sequences akiyo_cif,bus_cif
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.video import config
from benchmark.video.codecs import catalog
from benchmark.video.repro import build_metadata
from benchmark.video.stage1_extract import extract_contour_video

DATASET_NAME = "Xiph-CIF-natural"
# formal 与 speed 结果分文件写,避免 key 冲突 + 互染。
RESULTS_FILE_BY_MODE = {
    "formal": config.RESULTS_DIR / "xiph_cif.json",
    "speed": config.RESULTS_DIR / "xiph_cif_speed.json",
}

# Per-codec CRF/quality sweep — single source of truth: benchmark.video.codecs.catalog().
# (traditional → CRF 18-33; ssf2020 → q1,3,5,7,9; img-* → their qualities; dcvc_rt → qp.)
CODEC_CRF_MAP: dict[str, list[int]] = {c["id"]: list(c["qualities"]) for c in catalog()}
# Which codecs are traditional ffmpeg (CRF-sweepable) vs learned (quality-mapped).
_LEARNED_IDS = {c["id"] for c in catalog() if c["kind"] != "codec"}

PYTHON = sys.executable


def load_existing(results_file) -> list[dict]:
    if not results_file.exists():
        return []
    return json.loads(results_file.read_text(encoding="utf-8")).get("runs", [])


def key_of(run: dict) -> str:
    return f"{run.get('sequence_name')}|{run.get('codec')}|crf{run.get('crf')}"


def save(runs: list[dict], artifacts, codecs_used, crfs_used, frames, mode, results_file):
    meta = build_metadata(
        inputs=[str(a["manifest"]) for a in artifacts],
        codecs=codecs_used, crfs=sorted(crfs_used),
        method="canny", frame_cap=frames,
        runner="scripts/run_all_subprocess.py", dataset=DATASET_NAME,
    )
    payload = {"generated_at": datetime.now().isoformat(), "runs": runs, "mode": mode}
    payload.update(meta)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--sequences", default=None, help="comma-separated subset")
    ap.add_argument("--codecs", default=None, help="comma-separated subset")
    ap.add_argument("--mode", default="formal", choices=["formal", "speed"],
                    help="formal→xiph_cif.json, speed→xiph_cif_speed.json(分文件)")
    ap.add_argument("--crfs", default=None,
                    help="comma-separated CRF list; overrides the default sweep for "
                         "TRADITIONAL codecs only (learned keep their quality map)")
    args = ap.parse_args()

    config.ensure_dirs()
    env = {**os.environ, "NO_PROXY": "*", "no_proxy": "*", "PYTHONUTF8": "1"}

    results_file = RESULTS_FILE_BY_MODE[args.mode]

    # ---- Stage 1: extract each sequence once ----
    raw_dir = config.DATASETS_DIR / "raw" / "xiph_cif"
    seqs = sorted(raw_dir.glob("*.y4m"))
    if args.sequences:
        want = {s.strip() for s in args.sequences.split(",")}
        seqs = [s for s in seqs if s.stem in want]
    artifacts = []
    for y4m in seqs:
        print(f"[stage1] {y4m.name} (frames={args.frames})", flush=True)
        art = extract_contour_video(str(y4m), method="canny", frames=args.frames)
        artifacts.append({"stem": y4m.stem, "contour": str(art.frames_dir),
                          "manifest": str(art.manifest_path)})
        print(f"[stage1] {y4m.name}: {art.frame_count} frames", flush=True)

    codecs = list(CODEC_CRF_MAP)
    if args.codecs:
        want = {c.strip() for c in args.codecs.split(",")}
        codecs = [c for c in codecs if c in want]

    # --crfs overrides the CRF sweep for traditional codecs only; learned codecs
    # keep their quality map (CRF semantics differ — q1-9 vs CRF 0-63).
    if args.crfs:
        override = [int(c) for c in args.crfs.split(",") if c.strip()]
        for cid in codecs:
            if cid not in _LEARNED_IDS:
                CODEC_CRF_MAP[cid] = override

    runs = load_existing(results_file)
    have = {key_of(r) for r in runs}
    codecs_used, crfs_used = [], []
    print(f"[init] {len(runs)} existing runs; sweeping {len(codecs)} codecs x "
          f"{len(artifacts)} seqs", flush=True)

    for art in artifacts:
        for codec in codecs:
            crfs = CODEC_CRF_MAP[codec]
            pending = [c for c in crfs if f"{art['stem']}|{codec}|crf{c}" not in have]
            if not pending:
                print(f"  SKIP {art['stem']}|{codec}: all crfs exist", flush=True)
                continue
            cmd = [PYTHON, "-u", str(PROJECT_ROOT / "scripts" / "bench_one.py"),
                   "--sequence", art["stem"], "--contour-dir", art["contour"],
                   "--codec", codec, "--crfs", ",".join(str(c) for c in pending)]
            if args.frames is not None:
                cmd += ["--frames", str(args.frames)]
            print(f"  RUN  {art['stem']}|{codec} crfs={pending}", flush=True)
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
            except Exception as e:  # noqa: BLE001
                print(f"  SPAWN-FAIL {art['stem']}|{codec}: {e}", flush=True)
                continue
            for line in proc.stdout.splitlines():
                if line.startswith("RESULT:"):
                    try:
                        d = json.loads(line[len("RESULT:"):])
                        runs.append(d)
                        have.add(key_of(d))
                        if codec not in codecs_used:
                            codecs_used.append(codec)
                        if d.get("crf") not in crfs_used:
                            crfs_used.append(d["crf"])
                        print(f"    OK   {d['id']} PSNR={d.get('psnr',0):.2f} "
                              f"SSIM={d.get('ssim',0):.4f}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"    BADJSON: {e}", flush=True)
                elif line.startswith("ERROR:"):
                    try:
                        d = json.loads(line[len("ERROR:"):])
                        print(f"    ERR  {d.get('id')}: {d.get('error','')}", flush=True)
                    except Exception:
                        print(f"    {line}", flush=True)
            if proc.returncode != 0:
                # A segfault (139) here means bench_one crashed mid-run; any
                # RESULT lines emitted before the crash were already captured.
                print(f"    CRASH {art['stem']}|{codec} exit={proc.returncode} "
                      f"(partial results kept)", flush=True)
                if proc.stderr:
                    tail = proc.stderr.strip().splitlines()[-3:]
                    for tl in tail:
                        print(f"      stderr: {tl}", flush=True)
            save(runs, artifacts, codecs_used, crfs_used, args.frames, args.mode, results_file)

    save(runs, artifacts, codecs_used, crfs_used, args.frames, args.mode, results_file)
    print(f"[done] {len(runs)} runs -> {results_file}", flush=True)
    print(f"[done] {len(runs)} runs -> {results_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
