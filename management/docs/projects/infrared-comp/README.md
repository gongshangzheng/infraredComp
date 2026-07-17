---
title: 红外图像压缩
status: active
startDate: 2026-06-23
endDate:
tags: [红外压缩, 轮廓视频, 评测, 全栈, 数据集, 学习式 codec]
participants:
  - name: 郑鑫裕
    role: 全栈工程师
---

# 红外图像压缩

构建红外图像/视频压缩的评测与基准体系。前后端解耦:FastAPI 后端只读持久化数据(`management/` + `results/`),Vue3 前端展示。

## 评测管线(`benchmark/video/`)

两阶段:
- **阶段1 提取轮廓帧**:可插拔 `@register` 提取器(canny / sobel / hed / **pidinet / yoloe26**),产无损 `contour.mp4`(`libx264 -qp 0 yuv420p`)
- **阶段2 codec 压缩评测**:x264 / x265 / svtav1 / vp9 + 学习式(ssf2020 / dcvc_rt / img-*;接入中:lossless-seg / diffTok / DCVC 系列 / NEVC),出 PSNR / SSIM / 码率 / BD-Rate

## 数据集

| 数据集 | 用途 | 位置 |
|---|---|---|
| Xiph derf CIF | 自然视频 baseline | `datasets/raw/xiph_cif` |
| OSU Color-Thermal | 热红外视频 baseline | `datasets/raw/osu_color_thermal` |
| FLIR ADAS 1.3 | 红外图像(legacy) | `datasets/FLIR_ADAS_1_3` |
| **BSDS500** | 边缘检测 gt + diffTok 训练 | `D:/data/BSDS500` → `datasets/BSDS500`(junction) |
| **SA-Co-VEval** | 视频分割评测(Meta,~31GB) | `D:/data/SACo-VEval` → `datasets/SACo-VEval`(junction) |

大数据集(BSDS500 / SA-Co)走 `D:/data + symlink` 策略:真实数据在 `D:/data`(仓库外、不进 git),junction 到 `datasets/`(见 `dataset-management` skill)。

## 当前任务方向(详见 `tasks.json`)

- **评测提取器**:t7 PiDiNet / yoloe26(stage1,BSDS500+SA-Co 验证)
- **评测 codec**:t8 LosslessSeg、t10 DCVC 系列(6 成员)、t12 NEVC、t9 diffTok(stage2,SA-Co 评测)
- **训练**:t9 diffTok 灰度 enc/dec(BSDS500 训练)、t10 DCVC 系列
- **数据集**:t11 SA-Co-VEval(下载完成✅)、t13 BSDS500 gt 转 PNG
- **调研**:t5 3D 轮廓算子、t6 生成式压缩

任务单源 `management/docs/projects/infrared-comp/tasks.json`(看板 / 项目树页同源,`tasks_parser` 按 status 展平成 3 桶)。

## 仓库结构

`benchmark/video/`(评测库)+ `server/`(FastAPI)+ `web/`(Vue3)+ `management/`(项目数据)+ `datasets/`(junction 到 `D:/data` 大数据集)+ `models/`(权重 / vendor 模型)+ `scripts/`(下载 / baseline / 预览)。
