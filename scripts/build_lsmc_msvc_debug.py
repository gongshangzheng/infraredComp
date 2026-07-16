"""Rebuild LSMC with MSVC Debug (/Od, no optimization) — tests if the multi-label
segfault is caused by the MSVC /O2 optimizer (UB) vs a genuine code bug.

Usage: python scripts/build_lsmc_msvc_debug.py
"""
import subprocess, sys, os, tempfile, shutil
from pathlib import Path
import numpy as np

SRC = Path(tempfile.gettempdir()) / "lsmc-gcc"
DEST = Path(__file__).resolve().parent.parent / "third_party" / "lsmc"

def fix_cmake():
    p = SRC / "CMakeLists.txt"
    s = p.read_text(encoding="utf-8")
    s = s.replace("source/third_party/arithmetic_codec.cpp",
                  "source/arithmetic_coder/third_party/arithmetic_codec.cpp")
    if "acodec.cpp" not in s:
        s = s.replace("source/arithmetic_coder/third_party/arithmetic_codec.cpp",
                      "source/arithmetic_coder/third_party/arithmetic_codec.cpp\n    source/arithmetic_coder/acodec.cpp")
    s = s.replace("source/third_party", "source/arithmetic_coder/third_party")
    s = s.replace("      source/arithmetic_coder/third_party",
                  "      source/arithmetic_coder/third_party\n      source/arithmetic_coder")
    p.write_text(s, encoding="utf-8")
    print("[fix] CMakeLists.txt patched")

def main():
    if not SRC.is_dir():
        print(f"clone not found at {SRC}")
        return 1
    fix_cmake()
    build_dir = SRC / "build_dbg"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    print("[cmake] configure Debug")
    r = subprocess.run([
        "cmake", "-B", str(build_dir), "-S", str(SRC),
        "-G", "Visual Studio 17 2022", "-A", "x64",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  cmake configure FAIL: {r.stderr[-500:]}")
        return 1
    print("[build] Debug (no /O2 optimization)")
    r = subprocess.run(["cmake", "--build", str(build_dir), "--config", "Debug"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  build FAIL: {r.stderr[-500:]}")
        return 1
    enc = build_dir / "Debug" / "encoder.exe"
    print(f"[copy] {enc} -> {DEST / 'encoder.exe'}")
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(enc, DEST / "encoder.exe")
    # also copy decoder
    dec = build_dir / "Debug" / "decoder.exe"
    if dec.exists():
        shutil.copy2(dec, DEST / "decoder.exe")
    # test multi-label
    tmp = Path(tempfile.gettempdir())
    np.array([0,255,0,255, 255,0,255,0, 0,255,0,255, 255,0,255,0], dtype=np.uint8).tofile(str(tmp / "ml4.yuv"))
    r = subprocess.run([str(DEST / "encoder.exe"), "-i", str(tmp / "ml4.yuv"), "-o", str(tmp / "ml4.bin"),
                        "-r", "4", "-c", "4", "-f", "1", "-s", "0", "-t", "400"],
                       capture_output=True, text=True)
    print(f"[test] 4x4 multi-label (Debug /Od): exit {r.returncode}")
    if r.returncode == 0:
        print("  ✅ Debug build works on multi-label! The crash was MSVC /O2 optimizer UB.")
    else:
        print(f"  ❌ Still crashes in Debug too — genuine C++ bug, not optimizer.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
