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
