---
title: 红外图像压缩
status: active
startDate: 2026-06-23
endDate:
tags: [红外压缩, 轮廓视频, 评测, 全栈]
participants:
  - name: 张三
    role: 算法工程师
  - name: 李四
    role: 全栈工程师
---

# 红外图像压缩

本项目构建红外图像/视频压缩的评测与基准体系，覆盖轮廓视频两阶段压缩
（阶段1 提取轮廓帧、阶段2 用标准视频 codec 压缩并评测）、标准 codec
压缩评测、论文搜集与精读笔记等方向。前后端解耦：FastAPI 后端只读
持久化数据，Vue3 前端展示。
