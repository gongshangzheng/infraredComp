# 调研:图片生成式模型跑视频压缩

> 任务 t6 | 负责人:郑鑫裕 | 状态:planning | 2026-07-14

## 背景

当前 video benchmark 的 codec 是**传统视频 codec**(x264/x265/svtav1/vp9)+ **学习视频 codec**(ssf2020 CompressAI / dcvc-rt CVPR2025)。另一条路:**用已训练好的图片生成式模型逐帧压缩视频**,看效果(每帧当图片压缩,无时序预测,但用生成式/扩散类模型的高压缩比)。

## 调研方向

1. **生成式图像压缩**:Diffusion-based 厽名压缩(PNP, GAN-based, ELIC 等)—— 高感知质量、低码率,逐帧跑视频
2. **CompressAI 图像模型当视频用**:`bmshj2018 / mbt2018 / cheng2020 / ELIC` 逐帧压缩(已有 `benchmark/learned.py` + legacy 图像 benchmark),评估接入 stage2 in-process path
3. **新生成式模型**:查 SOTA 生成式压缩(MS-ILLM, DiffE, 等),预训练权重可用性
4. **效果维度**:PSNR/SSIM(客观)+ 感知质量(LPIPS/主观);与视频 codec(ssf2020/dcvc-rt,有时序)对比 —— 预期:无时序,码率可能高,但生成式感知质量可能优

## 接入点

`benchmark/video/codecs/base.py` 的 `VideoCodec` 有 **neural in-process path**(`is_neural=True`,`encode_inprocess(frames) -> bytes`):
- 仿 `ssf2020.py`:加载图片生成式模型,逐帧 `encode`(无 inter 帧预测),bitstream = 序列化各帧 bytes
- `decode_inprocess` 逐帧 decode
- metrics 走共享 PSNR/SSIM pipeline

## 待办

- [ ] 调研生成式图像压缩模型 + 预训练权重(网络:compressai S3 已确认 curl 可达)
- [ ] 仿 ssf2020.py 写图片生成式视频 codec(逐帧 in-process)
- [ ] 跑 baseline,对比 ssf2020(有时序)/ dcvc-rt / 传统 codec
- [ ] 评估感知质量(主观 + LPIPS 若可用)
