#!/usr/bin/env python3
"""等 imagenet hed 预提取完成，然后 detached 启动 ELIC 训练（用预提取 PNG，run_id 带 hed）。

预提取产物 manifest 的 frame_count 在跑完时跳到全量；本脚本轮询它到目标后启动训练，
训练用 DETACHED_PROCESS 起独立进程（脱离本脚本/会话）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

MF = REPO / "datasets" / "contour" / "imagenet_train_hed" / "manifest.json"
TARGET = 1281167  # imagenet-train 全量行数

BS = 96
NW = 4
EPOCHS = 100


def main() -> int:
    print("[orch] waiting for hed pre-extract to reach", TARGET, flush=True)
    last = 0
    while True:
        try:
            fc = json.load(open(MF, encoding="utf-8")).get("frame_count", 0)
        except Exception:
            fc = 0
        if fc >= TARGET:
            break
        if fc != last:
            print(f"[orch] pre-extract progress: manifest frame_count={fc} (target {TARGET})", flush=True)
            last = fc
        time.sleep(60)
    print(f"[orch] pre-extract DONE: {fc} frames", flush=True)

    rid = f"ELIC__hed__{int(time.time())}"
    logf = REPO / "results" / "training" / "logs" / f"{rid}.stdout.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    cmd = [
        sys.executable, "scripts/train_model.py",
        "--model", "ELIC", "--quality", "1", "--dataset", "imagenet-train",
        "--epochs", str(EPOCHS), "--lr", "1e-4", "--batch", str(BS), "--lambda", "0.01",
        "--device", "cuda", "--method", "hed", "--max-images", "0", "--shards", "0",
        "--num-workers", str(NW), "--size", "128", "--run-id", rid,
    ]
    subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT, creationflags=flags)
    print(f"[orch] launched ELIC training run_id={rid} bs={BS} nw={NW} (detached) -> {logf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
