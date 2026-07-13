---
name: compressai-usage
description: |
  CompressAI 库用法指南（infraredComp 视频压缩库）。说明 image + video 两套学习式压缩模型、zoo 注册表、compress/decompress API、训练 RD loss、checkpoint 加载（含 trained checkpoint override）、本库 benchmark/learned.py + benchmark/video/codecs/ + scripts/train_model.py 如何用 CompressAI。
  触发场景：(1) 加 CompressAI 模型（image/video）到 benchmark (2) 写 compress/decompress 编解码 (3) 训练/finetune CompressAI 模型 (4) checkpoint 加载/override (5) debug compress-decompress 不收敛
---

# CompressAI 库用法（infraredComp）

CompressAI（InterDigital）是学习式压缩库，infraredComp 用它做 **image** 学习压缩（`benchmark/learned.py`）+ **video** 学习压缩（`benchmark/video/codecs/ssf2020.py`，本库新增）。既用于**训练**（`scripts/train_model.py`）也用于**测试/推理**（benchmark eval）。本 skill 讲清 API + 本库模式。

## 库总览

```
compressai/
├── zoo/            # 模型工厂注册表
│   ├── image.py    # image_models: bmshj2018/cheng2020/mbt2018 ...（per-frame）
│   └── video.py    # video_models: {ssf2020}（时序帧序列）
├── models/
│   ├── priors.py bmshj2018.py cheng2020.py mbt2018.py google.py  # image 模型
│   └── video/google.py  # ScaleSpaceFlow (ssf2020)
├── entropy_models/ # EntropyBottleneck / GaussianConditional（熵编码 + likelihoods）
├── latent_codecs/  # 潜空间编解码
├── ops/            # bpp / distortion 辅助
└── registry.py     # @register_model / @register_metric
```

## image 模型（per-frame）

`compressai.zoo.image_models` 注册表：
```python
image_models = {
  "bmshj2018-factorized", "bmshj2018-factorized-relu", "bmshj2018-hyperprior",
  "mbt2018-mean", "mbt2018", "cheng2020-anchor", "cheng2020-attn",
  "bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr", "mbt2018-vbr",
}
# quality 1-8 (cheng2020 仅 1-6)
```

实例化 + compress/decompress：
```python
from compressai.zoo import image_models
model = image_models["bmshj2018-factorized"](quality=4, metric="mse", pretrained=True)
model = model.to(device).eval(); model.update()   # update() 必须在 compress/decompress 前调！

# 推理（实际压缩）
x = img_to_tensor(img)            # [1,3,H,W] float [0,1]
x = pad_to_multiple(x, 64)        # 神经 codec 需尺寸 ÷64
out = model.compress(x)           # {"strings": [bytes,...], "shape": (B,C,H,W)}
bits = sum(len(s) for s in out["strings"]) * 8
out_dec = model.decompress(out["strings"], out["shape"])  # {"x_hat": tensor}
x_hat = unpad(out_dec["x_hat"], original_hw)

# 训练（前向 + RD loss）
out = model(x)                    # {"x_hat": tensor, "likelihoods": dict|tensor}
x_hat, likelihoods = out["x_hat"], out["likelihoods"]
loss = lamb * bpp(likelihoods, num_pixels) + mse(x_hat, x)   # RD
```

## video 模型（时序帧序列）— `ssf2020`（ScaleSpaceFlow, Google CVPR2020）

`compressai.zoo.video_models = {"ssf2020": ssf2020}`，**CompressAI 唯一视频模型**，qualities 1-9，pretrained 在 `https://compressai.s3.amazonaws.com/models/v1/ssf2020-mse-{q}-*.pth.tar`。

```python
from compressai.zoo import video_models
model = video_models["ssf2020"](quality=5, metric="mse", pretrained=True)
model = model.to(device).eval(); model.update()

# frames = List[Tensor]  (注意是 Python list, 不是 batch tensor!)
# 每个 tensor [1,3,H,W] float [0,1]，H/W 需 ÷64

# 训练前向（返回重建 list + 每帧 likelihoods list）
out = model.forward(frames)      # {"x_hat": [t0_hat, t1_hat,...], "likelihoods": [lik0, lik1,...]}
# lik0 = {"keyframe": tensor};  lik_i(i>0) = {"motion": tensor, "residual": tensor}

# 实际压缩（编码 keyframe + 逐帧 inter）
x_hat_kf, out_kf = model.encode_keyframe(frames[0])     # out_kf = {"strings": [...], "shape": (B,C,H,W)}
x_ref = x_hat_kf
strings, shapes = [out_kf["strings"]], [out_kf["shape"]]
for f in frames[1:]:
    x_ref, out_i = model.encode_inter(f, x_ref)          # out_i = {"strings": {"motion":..,"residual":..}, "shape": {...}}
    strings.append(out_i["strings"]); shapes.append(out_i["shape"])

# 解码（镜像）
x_hat_kf = model.decode_keyframe(strings[0], shapes[0]); x_ref = x_hat_kf; recons = [x_hat_kf]
for i in range(1, len(frames)):
    x_ref = model.decode_inter(x_ref, strings[i], shapes[i]); recons.append(x_ref)
```

**bitstream 序列化**：`strings` 是 entropy-coded bytes 列表/dict，`shapes` 是 tensor 形状元数据。写文件时需自己序列化（pickle 或 自定义 framing：`[n_frames][per-frame: strings+shapes]`）。本库 `benchmark/video/codecs/ssf2020.py` 封装此序列化。

**比特统计**：`bits = sum(len(s) for s in all_strings)*8`；bpp = bits / (num_pixels * n_frames)。

## 训练 API（RD loss）

CompressAI 的 `CompressionModel.forward` 返回 `{"x_hat", "likelihoods"}`。RD loss：
```python
# image (per-frame)
out = model(x)
bpp = sum(-torch.log2(ll + 1e-10).sum() for ll in likelihoods_iter) / num_pixels   # 本库 scripts/train_model.py 用此
# 或用 compressai.ops.bpp(likelihoods, num_pixels) 助手
distortion = torch.mean((out["x_hat"] - x)**2)      # MSE
loss = lamb * bpp + distortion                       # λ 大 → 高码率高质量

# video (聚合关键帧 + inter)
out = model.forward(frames)                            # likelihoods = [lik_kf, lik_inter1, ...]
total_bits = sum(bpp_one_frame(lik) * num_pixels for lik in out["likelihoods"])
bpp = total_bits / (num_pixels * len(frames))
distortion = sum(torch.mean((xh - x)**2) for xh, x in zip(out["x_hat"], frames)) / len(frames)
loss = lamb * bpp + distortion
```

`scripts/train_model.py`（image）+ 视频扩展（ssf2020/dcvc-rt）用此模式。**warm-start**：`_load_model(checkpoint_path=pretrained)` 初始化再 fine-tune。

## checkpoint 加载（含 trained override）

```python
# pretrained（CompressAI 标准）
model = image_models[name](quality=q, pretrained=True)   # 自动从 ~/.cache/torch/hub/checkpoints/ 读 .pth.tar
# 或 video: video_models["ssf2020"](quality=q, pretrained=True)

# 自训练 checkpoint override（本库 _load_model(checkpoint_path=...) 钩子）
model = image_models[name](quality=q, pretrained=False)               # fresh
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict)
model.eval(); model.update()
```

pretrained 缓存路径：`~/.cache/torch/hub/checkpoints/{fname}`（CompressAI `load_state_dict_from_url` 落这里）。本库 `_load_model`（`benchmark/learned.py`）pre-check 此路径，缺则 `RuntimeError` 不自动下载（避免阻塞）。下载用 `scripts/download_learned_checkpoints.py`。

## 本库用法映射

| 场景 | 文件 | 用法 |
|------|------|------|
| image 学习压缩评测 | `benchmark/learned.py` | `_load_model(name, quality, device, checkpoint_path=…)` + `compress_learned()` |
| image 训练 | `scripts/train_model.py`（image 路径） | CompressAI `image_models[name](pretrained=False)` + RD loss |
| **video 学习压缩评测** | `benchmark/video/codecs/ssf2020.py`（新） | `video_models["ssf2020"]` + `encode_keyframe/encode_inter` 序列化 |
| **video 训练** | `scripts/train_model.py`（video 路径） | `model.forward(frames)` + 聚合 RD loss + warm-start |
| checkpoint 下载 | `scripts/download_learned_checkpoints.py`（新） | 拉 ssf2020 q1-9 + DCVC-RT pretrained |
| external 时序 codec | `benchmark/video/codecs/dcvc_rt.py`（新） | 非 CompressAI，自有 model 类 + checkpoint，同 LearnedVideoCodec 接口 |

## 关键约定

- **`model.update()` 必须在 compress/decompress 前调**（EntropyBottleneck 需要更新量化表）。`_load_model` 已含。
- **pad 到 64 倍数**（神经 codec 要求；learned.py `_pad_to_multiple(x, 64)`）；recon 裁回原尺寸。
- **`weights_only=False`**：`torch.load` 读 CompressAI checkpoint（含非张量状态）。
- **video frames 是 `List[Tensor]`**（不是 batch tensor）；每帧 [1,3,H,W]。
- **轮廓数据 OOD**：CompressAI/DCVC-RT pretrained 在自然图像/视频训练，在**轮廓帧**上效果差 → 需 fine-tune（见 `scripts/train_model.py` 视频路径）。
- **推理设备**：`model.to(device)`；CPU 可跑但慢，视频时序更慢。

## 触发场景速查

- 加 CompressAI image 模型 → `image_models` 注册表 + `_load_model`。
- 加 ssf2020 视频 → `video_models["ssf2020"]` + `encode_keyframe/encode_inter`（见 ssf2020.py）。
- 训练 → `forward` + RD loss + `torch.save(model.state_dict())`。
- checkpoint override → `pretrained=False` + `load_state_dict` + `update()`。
- debug 不收敛 → 查 `model.update()` 调了没、pad 64、likelihoods 正、λ 量级、frames 是 List。
