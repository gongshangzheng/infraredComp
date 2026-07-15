# 任务规划与跟踪

## 进行中

| 任务 | 负责人 | 开始日期 | 截止日期 | 状态 | 备注 |
|------|--------|----------|----------|------|------|
| dark 主题 + breadcrumb 回退 + 论文刷新保位(t4) | 郑鑫裕 | 2026-07-12 | 2026-07-13 | 🟢 | theme store/暗主题 + tooltip 透明背景 + 论文列表刷新保位 |
| 轮廓提取接入 PiDiNet / yoloe26 (t7) | 郑鑫裕 | 2026-07-15 |  | 🟢 | 接入 PiDiNet、yoloe26 到 extractors,与 canny/hed/sobel 并列 |
| 模型接入 LosslessSegmentationMapCompression (t8) | 郑鑫裕 | 2026-07-15 |  | 🟢 | InterDigitalInc/LosslessSegmentationMapCompression,接入为 codec/方法 |
| diffTok 灰度图 encoder/decoder (t9) | 郑鑫裕 | 2026-07-15 |  | 🟢 | 基于 diffTok 框架(gongshangzheng/diffTok),参考现有 enc/dec,实现灰度图版本 |

## 待开始

| 任务 | 负责人 | 预计开始 | 截止日期 | 优先级 | 备注 |
|------|--------|----------|----------|--------|------|
| 调研:视频 3D 轮廓提取算子(3D Sobel/3D Canny)(t5) | 郑鑫裕 | 2026-07-14 | — | 高 | 时空梯度算子,接入 extractors;详见 projects/infrared-comp/notes/03 |
| 调研:图片生成式模型跑视频压缩(t6) | 郑鑫裕 | 2026-07-14 | — | 高 | 生成式图像压缩逐帧跑视频;详见 projects/infrared-comp/notes/04 |

## 已完成

| 任务 | 负责人 | 完成日期 | 产出 | 备注 |
|------|--------|----------|------|------|
| 轮廓视频压缩评测库(t1) | 郑鑫裕 | 2026-07-11 | benchmark/video/ 两阶段 | 阶段1 轮廓提取 + 阶段2 codec 评测 |
| OSU Color-Thermal 数据集落地(t2) | 郑鑫裕 | 2026-07-11 | datasets/raw/osu_color_thermal + 下载脚本 | 注:vcipl-okstate.org 对部分网络 403,后改用 Xiph CIF |
| management CRUD skill + parser 空表修复(t3) | 郑鑫裕 | 2026-07-11 | .claude/skills/contour-video-management + markdown_table 修复 | task/member/meeting/report 全套 CRUD |

## 状态说明

- 🟢 正常推进
- 🟡 有风险 / 需关注
- 🔴 阻塞中
- ✅ 已完成
