"""Build LSMC encoder/decoder with MinGW g++ (not MSVC) — avoids the MSVC
multi-label segfault (C++ UB handled differently by g++ vs MSVC).

Usage: python scripts/build_lsmc_gcc.py
Expects: /tmp/lsmc-gcc (shallow clone of InterDigitalInc/LosslessSegmentationMapCompression)
Output: /tmp/lsmc-gcc/encoder_gcc.exe + decoder_gcc.exe → copied to models/lsmc/
"""
import subprocess, sys, os, shutil, tempfile
from pathlib import Path

GPP = r"C:\Users\wo\.conda\envs\compression\Library\mingw-w64\bin\g++.exe"
SRC = Path(tempfile.gettempdir()) / "lsmc-gcc"  # Windows %TEMP%\lsmc-gcc
DEST = Path(__file__).resolve().parent.parent / "models" / "lsmc"

INCLUDES = [
    "-Isource/arithmetic_coder",
    "-Isource/arithmetic_coder/third_party",
    "-Isource/commonlib",
    "-Isource/encoder",
    "-Isource/decoder",
]
COMMON = [
    "source/commonlib/coding_unit.cpp",
    "source/commonlib/global_arithmetic.cpp",
    "source/commonlib/utility.cpp",
    "source/arithmetic_coder/acodec.cpp",
    "source/arithmetic_coder/third_party/arithmetic_codec.cpp",
]
FLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra"]

def build(name, extra_sources):
    cmd = [GPP] + FLAGS + INCLUDES + COMMON + extra_sources + ["-o", f"{name}_gcc.exe", "-lstdc++"]
    print(f"[build] {name}")
    r = subprocess.run(cmd, cwd=str(SRC), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAIL: {r.stderr[-1500:]}")
        return False
    print(f"  OK: {name}_gcc.exe")
    return True

def main():
    if not SRC.is_dir():
        print(f"clone not found at {SRC}; run: git clone --depth 1 https://github.com/InterDigitalInc/LosslessSegmentationMapCompression.git /tmp/lsmc-gcc")
        return 1
    ok = True
    ok &= build("encoder", ["source/encoder/encoder.cpp", "source/encoder/encOptions.cpp"])
    ok &= build("decoder", ["source/decoder/decoder.cpp", "source/decoder/decOptions.cpp"])
    if not ok:
        print("BUILD FAILED")
        return 1
    # copy to models/lsmc/
    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("encoder", "decoder"):
        src = SRC / f"{name}_gcc.exe"
        dst = DEST / f"{name}.exe"
        shutil.copy2(src, dst)
        print(f"[copy] {src} -> {dst} ({dst.stat().st_size} bytes)")
    # test multi-label
    import numpy as np
    tmp = Path(tempfile.gettempdir())
    np.array([0,255,0,255, 255,0,255,0, 0,255,0,255, 255,0,255,0], dtype=np.uint8).tofile(str(tmp / "ml4.yuv"))
    r = subprocess.run([str(DEST / "encoder.exe"), "-i", str(tmp / "ml4.yuv"), "-o", str(tmp / "ml4.bin"),
                        "-r", "4", "-c", "4", "-f", "1", "-s", "0", "-t", "400"],
                       capture_output=True, text=True)
    print(f"[test] 4x4 multi-label encoder exit: {r.returncode}")
    if r.returncode == 0:
        print("  ✅ multi-label works with g++ build! No segfault!")
    else:
        print(f"  ❌ still crashes (exit {r.returncode})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
