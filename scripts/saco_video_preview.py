"""SA-Co video preview: triptych — orig | colored instance contours | white-on-black edges
(all mask boundaries merged, single white, instance/semantic-agnostic; same format as
BSDS500 gt boundaries / canny / hed / yoloe26 outputs for cross-comparison).

Decodes annotation masks via pycocotools, pipes raw frames to ffmpeg -> libx264 mp4
(cv2.VideoWriter writes empty mp4 due to openh264 dll mismatch, so use ffmpeg pipe).

Usage:
  PYTHONUTF8=1 /c/Users/wo/.conda/envs/compression/python.exe scripts/saco_video_preview.py [video] [split]
  default: saco_sg_000008 smartglasses_val
"""
import json, subprocess, sys
import numpy as np, cv2
from pathlib import Path
from pycocotools import mask as M

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.video.config import raw_dir  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "saco_sg_000008"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "smartglasses_val"
FF = r'D:\code\infraredComp\.venv\Lib\site-packages\static_ffmpeg\bin\win32\ffmpeg.exe'

JSON_P = raw_dir("SACo-VEval") / f"annotation/saco_veval_{SPLIT}.json"
FRAME_DIR = raw_dir("SACo-VEval") / f"media/saco_sg/JPEGImages_6fps/{VIDEO}"
OUT = Path("results/video/saco_preview"); OUT.mkdir(parents=True, exist_ok=True)

d = json.load(open(JSON_P, encoding="utf-8"))
vid = [v for v in d["videos"] if v["video_name"] == VIDEO][0]
anns = [a for a in d["annotations"] if a["video_id"] == vid["id"]]
print(f"{VIDEO}: {len(vid['file_names'])} frames, {len(anns)} objects")

def decode(rle):
    h, w = rle["size"]; c = rle["counts"]
    if isinstance(c, str): c = c.encode("ascii")
    return M.decode({"counts": c, "size": [h, w]})

cols = [(0,255,255),(0,255,0),(255,0,255),(255,255,0),(0,165,255),(255,0,0),(128,255,255),(255,128,0)]

f0 = cv2.imread(str(FRAME_DIR / Path(vid["file_names"][0]).name))
H, W = f0.shape[:2]
out_w = W * 3 + 16  # img(W) + gap(8) + overlay(W) + gap(8) + edges(W)
mp4 = OUT / f"{VIDEO}.mp4"
cmd = [FF, '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{out_w}x{H}',
       '-framerate', '6', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
       '-preset', 'fast', str(mp4)]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
n = 0
for fidx, fn in enumerate(vid["file_names"]):
    img = cv2.imread(str(FRAME_DIR / Path(fn).name))
    if img is None: continue
    overlay = img.copy()
    edges = np.zeros((H, W, 3), np.uint8)  # black bg, white contours (instance-agnostic)
    for ai, a in enumerate(anns):
        seg = a["segmentations"][fidx]
        if seg is None: continue
        m = decode(seg)
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = cols[ai % len(cols)]
        cv2.drawContours(overlay, cnts, -1, col, 3)            # colored instance contours
        cv2.drawContours(edges, cnts, -1, (255, 255, 255), 2)  # white, instance-agnostic
        if cnts:
            Mm = cv2.moments(cnts[0])
            if Mm["m00"] > 0:
                cx = int(Mm["m10"] / Mm["m00"]); cy = int(Mm["m01"] / Mm["m00"])
                cv2.putText(overlay, a.get("noun_phrase", "")[:18], (max(0, cx-40), max(15, cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    gap = np.full((H, 8, 3), 255, np.uint8)
    frame = np.hstack([img, gap, overlay, gap, edges])
    assert frame.shape[1] == out_w, (frame.shape, out_w)
    p.stdin.write(frame.tobytes()); n += 1
p.stdin.close(); err = p.stderr.read().decode('utf-8', 'replace'); rc = p.wait()
print(f"ffmpeg rc={rc}, wrote {n} frames, frame_w={out_w} -> {mp4} ({mp4.stat().st_size if mp4.exists() else 0} bytes)")
if rc: print("stderr:", err[-1200:])
