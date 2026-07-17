# DCVC-RT 接入问题记录

## 问题1: MLCodec_extensions_cpp Windows crash (exit 127)
encode_z+encode_y x2+ get_encoded_stream crash. /Od 也崩.
修复: 重写 EntropyCoder 用 MLCodec_rans (DCVC-DC ext):
- add_cdf → _cdf_store dict
- encode_y → 解包 int16 → encode_with_indexes
- encode_z/decode_z → arange(channel)+start_offset tiled indexes
- get_y → decode_stream(filtered) → masked_scatter scatter-back

## 问题2: CUDA stream crash
patch image_model.py + video_model.py → default stream

## 问题3: 2nd ext build 失败 (nvcc C2872)
未解, fallback pytorch 可用

## 问题4: setup.py /WX + C4819
加 /utf-8, 去 /WX

## 验证: bpp=1.85 psnr=12.05 bytes=93560 (4帧 crf=20)

---

## 2026-07-17 复核与修正（原版 EntropyCoder 路线打通）

目标改为：让 `dcvc_rt` 用 DCVC **原版 `EntropyCoder`（`MLCodec_extensions_cpp`）** 并在 Windows 编译通过，**不走** `MLCodec_rans` 重写。

**问题1 已证伪（过时）**：当年报的 "encode_z+encode_y x2+get_encoded_stream crash exit 127, /Od 也崩" 复测不复现。
用 compression env 已装的 `.pyd` 跑独立 roundtrip（不依赖神经模型/checkpoint）：
- 单编码器：encode_y+encode_z+flush+get_encoded_stream+decode 全通，roundtrip 精确。
- 双编码器（`set_use_two_encoders(True)`，10 万符号）：同样 roundtrip 精确。
即原版 ext 的 rans 逻辑本身正常。当年 crash 大概率是 **问题2 CUDA stream** 的表象，问题1 的 `MLCodec_rans` 重写是过度反应（且 `MLCodec_rans` 属 DCVC-DC/NEVC 路线，`dcvc_rt.py` 本就没用它）。

**构建重新打通（问题4 patch 重新打上）**：submodule 重拉后 upstream setup.py 仍是原版 `/WX` 无 `/utf-8`，当年 patch 是未提交 working-tree 改动、随 deinit 丢失。重打：
- `models/DCVC/src/cpp/setup.py` win32 分支：`['/std:c++17','/O2','/W4','/utf-8','/wd4100']`（去 `/WX`，加 `/utf-8`）。
- 构建：`vcvars64.bat`（VS 2022 Community @ `D:\Program Files\Microsoft Visual Studio\2022\Community`）+ compression env（pybind11 3.0.4）+ `pip install . --no-build-isolation --force-reinstall --no-deps`（必须 `--no-build-isolation`，否则 build env 无 pybind11）。
- 产物 `MLCodec_extensions_cpp.cp312-win_amd64.pyd` 装回 compression env site-packages，D9025（pybind11 默认 `/std:c++latest /W3` 被覆盖）仅警告非错误。
- 验证：新 .pyd 双 repro 全过。`dcvc_rt._load()` 越过 repo+ext 检查，推进到 checkpoint 缺失步（预期，权重需手动下）。

**剩余 runtime blocker（非构建问题，需 checkpoint 才能测）**：
1. **CVPR-2025 checkpoint**（手动 OneDrive，不可脚本化）：`cvpr2025_image.pth.tar` + `cvpr2025_video.pth.tar` → `models/DCVC/checkpoints/`。
2. **问题2 CUDA default-stream patch**：submodule 现为干净 upstream，`image_model.py:168` / `video_model.py:326,365` 仍用 `torch.cuda.stream(cuda_stream)`（非默认流）。当年 patch 成默认流以让 `entropy_coder.encode_y(symbols.cpu().numpy(), ...)` 的 `.cpu()` 同步、避免 CUDA stream desync crash。需重新打上（改 `common_model.py` 的 stream 创建或那三处 `with torch.cuda.stream(...)`）。**无 checkpoint 无法验证**，故暂不盲改，待权重到位后小序列验证 encode 不崩再定稿。
3. 可选 fused ext `inference_extensions_cuda`（问题3 nvcc C2872 未解）——fallback pytorch 算子可跑，仅慢。
