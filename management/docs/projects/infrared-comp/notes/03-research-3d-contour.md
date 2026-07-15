# 调研:视频 3D 轮廓提取算子(3D Sobel / 3D Canny)

> 任务 t5 | 负责人:郑鑫裕 | 状态:planning | 2026-07-14

## 背景

当前 `benchmark/video/extractors/` 的 canny/sobel 是**帧级 2D**算子(`extract(frame_gray) -> uint8`),逐帧独立提取,不利用时序信息。对视频轮廓,3D 算子(时空梯度)可能更优(沿时间轴的边缘、运动边界)。

## 调研方向

1. **3D Sobel**:在 (x, y, t) 三维卷积 Sobel 核(或 3D 梯度幅值)。查找:
   - OpenCV / scikit-image 是否有 3D Sobel(`scipy.ndimage.sobel` 支持 axis,可组合 3D)
   - 经典论文/实现:3D edge detection in video
2. **3D Canny**:时空 Canny(非极大值抑制 + 滞后在 3D)。查找:
   - 是否有公开实现(通常 Canny 是 2D,3D 需自定义)
3. **其他时空边缘**:时空梯度(Lucas-Kanada 类)、运动边界(motion boundary)、optical flow 边缘

## 接入点

`benchmark/video/extractors/base.py` 的 `ContourExtractor.extract(frame_gray) -> uint8` 是**单帧接口**。3D 算子需要**帧序列**上下文(前后帧)。接入方式:
- 扩展 `ContourExtractor` 加 `extract_sequence(frames) -> list`(或 `extract(frame_gray, context)`)
- stage1 `extract_contour_video` 改为批量传帧序列
- 新提取器 `@register("3d_sobel")` / `@register("3d_canny")`

## 评估

- 3D 算子对**轮廓视频压缩**的 RD 影响(时序一致性 temporal_metric 可能改善)
- 计算开销(3D 卷积比 2D 重)
- 与 2D canny/sobel 的对比 baseline

## 待办

- [ ] 查 scikit-image / scipy / OpenCV 的 3D Sobel/Canny 可用性
- [ ] 读 3D edge detection 经典方法
- [ ] 扩展 extractor 接口支持序列
- [ ] 实现 3d_sobel,对比 2D baseline
