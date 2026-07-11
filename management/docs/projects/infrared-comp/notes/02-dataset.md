# OSU Color-Thermal 数据集

来自 OTCBVS Dataset 03（`https://vcipl-okstate.org/pbvs/bench/Data/03/download.html`），6 段真实热红外视频序列（320×240, 8-bit）。下载脚本 `scripts/download_osu_color_thermal.py` 自动下载 + 归一化为 `datasets/raw/osu_color_thermal/seq{1..6}.mp4`（h264/yuv420p/25fps）。

baseline 样本量说明：6 段够做方向性 baseline；正式结论建议扩到 ~12-16 段（补 BU-TIV / FLIR ADAS）。
