# 轮廓视频压缩评测

两阶段评测（`benchmark/video/`）：

- **阶段1**：从原始视频提取轮廓帧（`canny` / `sobel` 等可插拔提取器），产出无损灰度 PNG + `manifest.json`。
- **阶段2**：用标准视频 codec（`x264` / `x265` / `svtav1` / `vp9`）压缩轮廓视频，统一 `-pix_fmt yuv420p` + 奇数尺寸 pad，算 PSNR/SSIM/码率/BD-Rate。

```bash
uv run python -m benchmark.video --input datasets/raw/osu_color_thermal/seq1.mp4 \
  --method canny --crfs 18,23,28,33
```
