---
name: dcvc-rt-usage
description: |
  DCVC-RT（microsoft/DCVC, CVPR 2025 实时神经视频 codec）用法指南（infraredComp 视频压缩库）。说明 DMCI（I 帧）+ DMC（P 帧）两模型、compress/decompress API、rans C++ ext 强依赖、DPB seeding（clear_dpb + add_ref_frame 用 I-recon）、CVPR-2025 checkpoint 手动下载放置、CPU fp32 vs CUDA fp16、推理专用（无训练代码）。本库 benchmark/video/codecs/dcvc_rt.py 就是该 codec；checkpoint override 走 checkpoint_i/checkpoint_p。
  触发场景：(1) 在 eval 里用 DCVC-RT (2) debug compress/decompress (3) 装 rans ext (4) 放置 checkpoint (5) 理解 qp↔crf 语义。
---

# DCVC-RT 库用法（infraredComp）

DCVC-RT = microsoft/DCVC 顶层（CVPR 2025 实时神经视频 codec），MIT license。infraredComp 用它做 **video** 学习压缩（`benchmark/video/codecs/dcvc_rt.py`，本库新增），走 in-process 神经 codec 路径（无 ffmpeg）。**推理专用**——Microsoft 未放出训练代码，contour finetune 属未来自定义 RD-loss 训练工作。

## 库总览

```
third_party/DCVC/                # git submodule (git submodule update --init)
├── src/
│   ├── models/
│   │   ├── image_model.py    # DMCI（I 帧，intra）
│   │   ├── video_model.py    # DMC（P 帧，inter）— 内含 RefFrame + DPB
│   │   ├── common_model.py   # CompressionModel 基类：get_qp_num/get_padding_size/update/pad_for_y
│   │   └── entropy_models.py # EntropyCoder（调 MLCodec_extensions_cpp 的 RansEncoder/Decoder）
│   ├── layers/
│   │   └── cuda_inference.py # replicate_pad / CUSTOMIZED_CUDA_INFERENCE / round_and_to_int8
│   │   └── extensions/inference/  # 可选 CUDA fused ext（inference_extensions_cuda）
│   ├── utils/common.py       # get_state_dict（剥 module./state_dict/net 前缀）
│   └── cpp/py_rans/           # rans C++ ext 源码（build → MLCodec_extensions_cpp）
├── checkpoints/              # 手动放 CVPR-2025 权重（OneDrive 下载）
│   ├── cvpr2025_image.pth.tar
│   └── cvpr2025_video.pth.tar
└── test_video.py             # 参考编解码 + DPB seeding（本 codec 实现的模板）
```

## 两模型 + API

| 模型 | 文件 | 帧类型 | compress 返回 | decompress 返回 |
|---|---|---|---|---|
| `DMCI` | `src/models/image_model.py` | I 帧 | `{"bit_stream": bytes, "x_hat": tensor}` | `{"x_hat": tensor}` |
| `DMC`  | `src/models/video_model.py` | P 帧 | `{"bit_stream": bytes}`（**无 x_hat**） | `{"x_hat": tensor}` |

```python
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import get_state_dict

i_net = DMCI(); i_net.load_state_dict(get_state_dict(ckpt_i))
p_net = DMC();  p_net.load_state_dict(get_state_dict(ckpt_p))
for net in (i_net, p_net):
    net = net.to(device).eval()
    net.update(force_zero_thres=0.12)   # README 推荐值；初始化 entropy_coder，compress/decompress 前必调

# --- compress ---
enc_i = i_net.compress(x_padded, qp)        # x_padded: (1,3,H,W) float [0,1]，÷16 pad 后
# bit_stream = enc_i["bit_stream"]  (rans 实算术编码 bytes)
# x_hat      = enc_i["x_hat"]       (I-recon，用于 seed P-DPB)

# DPB seeding：用 I-recon 作为 P 帧的参考帧（frame 而非 feature）
p_net.clear_dpb()
p_net.add_ref_frame(None, enc_i["x_hat"])   # feature=None, frame=I-recon

enc_p = p_net.compress(x_padded, qp)        # DMC.compress 内部自动 add_ref_frame(feature, None)
# 后续 P 帧：DPB 由 DMC 自己推进，不用再手动 seed

# --- decompress ---
sps = {"sps_id": 0, "height": h, "width": w, "ec_part": ec, "use_ada_i": 0}
dec_i = i_net.decompress(bit_stream_i, sps, qp)
p_net.clear_dpb()
p_net.add_ref_frame(None, dec_i["x_hat"])   # decode 端同样 seed I-recon
dec_p = p_net.decompress(bit_stream_p, sps, qp)  # DMC.decompress 内部自动 add_ref_frame
x_hat = dec_p["x_hat"][:, :, :h, :w]         # crop ÷16 padding
```

关键点：
- **qp 范围 0..63**（`DMC.get_qp_num()==64`），**qp 越低 = 质量越高**。
- `bit_stream` 是 **rans C++ 扩展产生的真实算术编码 bytes**（不是 CompressAI 的 strings 列表），必须 build `MLCodec_extensions_cpp`，**无 fallback**。
- **DPB seeding**：每个 I 帧后 `clear_dpb()` + `add_ref_frame(None, x_hat)`；P 帧 `compress`/`decompress` 内部自己 `add_ref_frame`，DPB 自动推进。decode 端必须镜像 seed，否则 P 帧无参考。
- `sps` 需要 `height/width`（**未 pad 的原始尺寸**，decompress 内部按 64/16 下采样算 z_size/y_size）、`ec_part`（>1280×720 用双熵编码器）、`use_ada_i`（本 codec 固定 0，不跑 feature-adaptor-i reset）。

## 本库 codec：`benchmark/video/codecs/dcvc_rt.py`

`@register_codec("dcvc_rt")`，`family="learned-video"`，`ext="bin"`，`is_neural=True`。

```python
from benchmark.video.codecs import build_codec
codec = build_codec("dcvc_rt", crf=23)   # crf 直接当 qp：qp=clamp(crf,0,63)
# qp = 23（crf→qp 1:1；higher crf = lower quality = fewer bits，与 x264 sweep 同向）
bs: bytes = codec.encode_inprocess(frames, fps)              # frames=list[HxW uint8]
rec: list = codec.decode_inprocess(bs, n_frames, (h, w))     # -> list[HxW uint8]
```

### crf↔qp 语义（重要）
本 benchmark 默认 `crfs=[18,23,28,33]` 是 **x264 风格**（higher crf = lower quality = fewer bits）。DCVC qp 0=最高质量、63=最低。为保证 RD 曲线方向与 x264/x265/svtav1/vp9 **一致**，本 codec 直接 `qp = clamp(crf, 0, 63)`（1:1）。注意这跟 `ssf2020` **相反**（ssf2020 的 crf=quality 1-9，higher=more bits）——同一 crf 值下两 codec 不可直接比较。

### 帧序列化格式（binary container，非 pickle）
```
magic(8)="DCVCRT10" + n(uint32) + h(uint32) + w(uint32)
per frame: type(uint8, 0=I/1=P) + qp(int32) + blen(uint32) + bit_stream(blen bytes)
stats_len(uint32) + pickle(stats_list)   # 每帧 min-max norm 的 (img_min,img_max)，decode 反 norm 用
```
每帧 I/O 复用 `benchmark/learned.py` 的 `_img_to_tensor`（gray→3ch + min-max norm 到 [0,1]）/`_tensor_to_img`（反 norm 回 uint8）；per-frame stats 序列化进 bitstream（decode 无原始帧也能反 norm）。

### padding：÷16（不是 ÷64）
DCVC-RT 用 `DMCI.get_padding_size(h, w, 16)` + `replicate_pad(x, pad_b, pad_r)`（注意参数顺序 bottom, right），来自 `src/layers/cuda_inference`。与 ssf2020 的 ÷64 不同——别混用。

### dtype / 设备
- **CPU**：模型 fp32，输入 fp32（**跳过 `.half()`**）。
- **CUDA**：模型 `.half()`（fp16，更快），输入也 `.half()`。

### checkpoint override
`DCVCRTCodec(crf, checkpoint_i=..., checkpoint_p=...)` 覆盖默认 `<repo>/checkpoints/cvpr2025_*.pth.tar`。**目前无 contour-finetuned checkpoint**（推理专用，无训练代码）；一旦未来训练出 contour-finetuned 权重，走这俩参数 eval。

## 安装（必须，且不自动）

`_load` 默认用项目内 **git submodule** `third_party/DCVC`（pin 版本，不靠外部 env）；rans ext 未 build / checkpoint 缺失时抛 **清晰 RuntimeError**（带步骤），**不在 import 时崩**。完整步骤：

1. **fetch DCVC submodule**（microsoft/DCVC，CVPR 2025 顶层，MIT）：
   ```bash
   git submodule update --init third_party/DCVC
   ```
   （也可 `export DCVC_REPO_ROOT=/path/to/DCVC` 覆盖指向自建 clone。）
2. **build rans C++ ext（必须，无 fallback）**：
   ```bash
   cd third_party/DCVC/src/cpp && pip install .
   # 装 MLCodec_extensions_cpp；需 cmake/g++/ninja + pybind11
   ```
3. **(可选, 仅 CUDA) fused ext**：
   ```bash
   cd third_party/DCVC/src/layers/extensions/inference && pip install .
   # 装 inference_extensions_cuda；不装则 CUSTOMIZED_CUDA_INFERENCE=False，自动 fallback 到 pytorch 算子（慢一点但能跑）
   ```
4. **下 CVPR-2025 checkpoint（手动，不可脚本化）**：
   - OneDrive 链接见 DCVC repo `README.md`（`*Download our pretrained models*`）。
   - 放到 `third_party/DCVC/checkpoints/`：
     - `cvpr2025_image.pth.tar`（DMCI, I 帧）
     - `cvpr2025_video.pth.tar`（DMC, P 帧）

## 跑 eval

```bash
# 装好上面 5 步后
uv run python -m benchmark.video --input <contour_artifact> --codecs dcvc_rt --crfs 18,23,28,33
```
不装 rans ext / checkpoint 直接跑 → `_load` 抛 RuntimeError 提示哪步缺。

## 限制 / 未来工作
- **推理专用**：Microsoft 未放出训练代码。contour finetune = 未来自定义 RD-loss 训练（参考 `learned-codec-install` skill 的训练接入流程，但 DCVC-RT 需自己实现 train loop + entropy coder 兼容）。
- **单 GOP / 无 intra refresh**：本 codec 用最简路径（constant qp，无 `shift_qp` 轮转、无 `use_ada_i` reset），短 contour 视频够用；长序列可考虑加 reset_interval（参考 `test_video.py` 的 `index_map`/`shift_qp`）。
- **CPU 慢**：DCVC-RT 实时性靠 CUDA fused ext；CPU fp32 仅用于 debug/小序列。

## 相关
- `learned-codec-install`：集成新学习压缩库的 meta-skill（DCVC-RT 已按此流程装好 + 本 skill 即其产物）。
- `compressai-usage`：兄弟学习视频 codec ssf2020 的 API（CompressAI，有训练代码，可 finetune）。
- `contour-video-evaluation`：stage2 benchmark 怎么调 codec（crf grid、recon PNG、PSNR/SSIM/bpp）。
