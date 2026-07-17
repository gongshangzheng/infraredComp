# -*- coding: utf-8 -*-
# Guide 红外相机 MJPEG 桥接: 帧回调拿 Y16 全帧 -> 百分位归一化 -> ironbow 伪彩 -> JPEG -> HTTP MJPEG.
# 浏览器打开 http://127.0.0.1:8080 看实时红外画面.  Ctrl+C 停.
import os, sys, ctypes, time, io, threading
from ctypes import c_int, c_void_p, POINTER, CFUNCTYPE, Structure, byref, string_at
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image
import cv2

SDK = r"D:\Programs\GuideIR_V1.2.4"
LIBUSB_DIR = os.path.join(SDK, "AllDriverAutoInstall", "amd64")
os.add_dll_directory(SDK); os.add_dll_directory(LIBUSB_DIR)
os.environ["PATH"] = LIBUSB_DIR + os.pathsep + SDK + os.pathsep + os.environ["PATH"]
os.chdir(SDK)

DEVID = 669823954
WIDTH, HEIGHT, VERSION = 1280, 1024, 3
YUV = 0
PORT = 8080
FPS = 15
DEFAULT_PALETTE = "ironbow"


class guide_usb_device_info_t(Structure):
    _fields_ = [("width", c_int), ("height", c_int), ("video_mode", c_int), ("device_version", c_int)]


class guide_usb_frame_data_t(Structure):
    _fields_ = [
        ("frame_width", c_int), ("frame_height", c_int),
        ("frame_rgb_data", c_void_p), ("frame_rgb_data_length", c_int),
        ("frame_src_data", c_void_p), ("frame_src_data_length", c_int),
        ("frame_yuv_data", c_void_p), ("frame_yuv_data_length", c_int),
        ("frame_param_data", c_void_p), ("frame_param_data_length", c_int),
    ]


OnFrameDataReceivedCB = CFUNCTYPE(None, POINTER(guide_usb_frame_data_t))
OnDeviceConnectStatusCB = CFUNCTYPE(None, c_int)

_lock = threading.Lock()
_latest = {"raw": None, "count": 0}


def on_frame(data_p):
    try:
        d = data_p.contents
        ptr = d.frame_src_data
        _latest["count"] += 1
        if ptr:
            raw = string_at(ptr, WIDTH * HEIGHT * 2)  # 全帧 Y16 字节
            with _lock:
                _latest["raw"] = raw
    except Exception as e:
        print(f"on_frame EXC: {e!r}", flush=True)


def on_status(s):
    print(f"[status] {s} ({'CONNECT_OK' if s == 1 else 'DISCONNECT' if s == -1 else s})", flush=True)


_frame_cb = OnFrameDataReceivedCB(on_frame)
_status_cb = OnDeviceConnectStatusCB(on_status)


def _build_luts():
    x = np.arange(256, dtype=np.float32) / 255.0
    iron = np.stack([np.interp(x, [0,.25,.5,.75,1],[0,.4,.9,1,1]),
                     np.interp(x, [0,.4,.7,1],[0,.1,.6,1]),
                     np.interp(x, [0,.2,.5,1],[0,.5,.1,1])], axis=1)
    rb = np.stack([np.interp(x,[0,.25,.5,.75,1],[0,0,0,1,1]),
                   np.interp(x,[0,.25,.5,.75,1],[0,0,1,1,0]),
                   np.interp(x,[0,.25,.5,1],[1,1,0,0])], axis=1)
    d = {"whitehot": np.stack([x]*3, axis=1),
         "blackhot": np.stack([1-x]*3, axis=1),
         "ironbow": iron, "rainbow": rb}
    return {k: (v*255).astype(np.uint8) for k, v in d.items()}


LUTS = _build_luts()


def render_jpeg(raw, palette=DEFAULT_PALETTE):
    y16 = np.frombuffer(raw, dtype=np.uint16).reshape(HEIGHT, WIDTH)
    lo, hi = np.percentile(y16, [1, 99])  # 百分位归一化, 解低对比度
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((y16.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)

    if palette in ("sobel", "canny"):
        gray = idx  # 8-bit 灰度
        if palette == "canny":
            edges = cv2.Canny(gray, 50, 150)
            rgb = np.stack([edges] * 3, axis=2)  # 边缘=白, 背景黑
        else:  # sobel
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.clip(np.sqrt(sx**2 + sy**2), 0, 255).astype(np.uint8)
            rgb = np.stack([mag] * 3, axis=2)
    else:
        lut = LUTS.get(palette, LUTS[DEFAULT_PALETTE])
        rgb = lut[idx]  # (H, W, 3)

    img = Image.fromarray(rgb, "RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=72)
    return buf.getvalue()


HTML = b"""<html><head><meta charset="utf-8"><title>Guide IR Live</title>
<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}
img{width:100vw;height:100vh;object-fit:contain}
.hint{position:fixed;top:6px;left:8px;font-size:12px;opacity:.7}</style></head>
<body><div class="hint">Guide IR MJPEG bridge</div><img src="/stream"></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML)
        elif urlparse(self.path).path == "/stream":
            q = parse_qs(urlparse(self.path).query)
            palette = q.get("palette", [DEFAULT_PALETTE])[0]
            if palette not in LUTS:
                palette = DEFAULT_PALETTE
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        raw = _latest["raw"]
                    if raw:
                        jpg = render_jpeg(raw, palette)
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1.0 / FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()


def main():
    cam = ctypes.WinDLL(os.path.join(SDK, "GuideUSBCamera.dll"))
    cam.guide_usb_initialize.restype = c_int; cam.guide_usb_initialize.argtypes = []
    cam.guide_usb_openStreamByDevID.restype = c_int
    cam.guide_usb_openStreamByDevID.argtypes = [c_int, POINTER(guide_usb_device_info_t),
                                                OnFrameDataReceivedCB, OnDeviceConnectStatusCB]
    cam.guide_usb_closeStream.restype = c_int; cam.guide_usb_closeStream.argtypes = []
    cam.guide_usb_exit.restype = c_int; cam.guide_usb_exit.argtypes = []

    rc = cam.guide_usb_initialize()
    print(f"initialize rc={rc}", flush=True)
    info = guide_usb_device_info_t(WIDTH, HEIGHT, YUV, VERSION)
    rc = cam.guide_usb_openStreamByDevID(DEVID, byref(info), _frame_cb, _status_cb)
    print(f"openStreamByDevID rc={rc}", flush=True)

    server = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    print(f"MJPEG server on http://127.0.0.1:{PORT}  (Ctrl+C stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("closing...", flush=True)
        cam.guide_usb_closeStream()
        cam.guide_usb_exit()
        print("done", flush=True)


if __name__ == "__main__":
    main()
