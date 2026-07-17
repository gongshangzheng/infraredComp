"""Video codecs. Catalog is STATIC (no codec module import here);
build_codec lazy-imports the specific codec module on first use.

This avoids loading torch/compressai/ultralytics just to list the catalog
(/evaluation/codecs). Registration (@register_codec) happens on lazy import.
"""

from .base import (  # noqa: F401
    CodecConfig,
    VideoCodec,
    CODEC_REGISTRY,
    build_codec,
    register_codec,
)

__all__ = [
    "CodecConfig",
    "VideoCodec",
    "CODEC_REGISTRY",
    "build_codec",
    "register_codec",
    "catalog",
    "list_codecs",
]


# ---- unified codec catalog (static — no import of codec modules) ----
# Single source of truth for /evaluation/codecs + runner CRF map.
# build_codec lazy-imports the codec module (register_codec) on first use.
# If a new codec is added, add its entry here AND its module path in
# base.py _CODEC_MODULE (or it'll be "img-*" → learned_image).
_CODEC_CATALOG: list[dict] = [
    {"id": "x264", "name": "x264 (H.264/AVC)", "family": "x264", "kind": "codec", "ext": "mp4", "is_neural": False, "qualities": [18, 23, 28, 33], "trainable": False, "description": "最通用基线 codec"},
    {"id": "x265", "name": "x265 (HEVC)", "family": "x265", "kind": "codec", "ext": "mp4", "is_neural": False, "qualities": [18, 23, 28, 33], "trainable": False, "description": "高压缩比现代 codec"},
    {"id": "svtav1", "name": "SVT-AV1", "family": "svtav1", "kind": "codec", "ext": "mp4", "is_neural": False, "qualities": [18, 23, 28, 33], "trainable": False, "description": "新一代 royalty-free，较慢（static-ffmpeg win32 无 libsvtav1）"},
    {"id": "vp9", "name": "VP9", "family": "vp9", "kind": "codec", "ext": "webm", "is_neural": False, "qualities": [18, 23, 28, 33], "trainable": False, "description": "Google 开源 codec"},
    {"id": "mpeg4", "name": "MPEG-4 Part 2", "family": "mpeg4", "kind": "codec", "ext": "mp4", "is_neural": False, "qualities": [8, 14, 20, 26], "trainable": False, "description": "DivX/Xvid 系老一代基线；qscale 控质量，码率效率低于 H.264"},
    {"id": "ssf2020", "name": "ssf2020 (Scale-Space Flow)", "family": "ssf2020", "kind": "learned-video", "ext": "bin", "is_neural": True, "qualities": [1, 3, 5, 7, 9], "trainable": True, "description": "CompressAI 视频模型 CVPR2020；可 fine-tune"},
    {"id": "dcvc_rt", "name": "DCVC-RT (real-time NVC)", "family": "dcvc_rt", "kind": "learned-video", "ext": "bin", "is_neural": True, "qualities": [20, 30, 40], "trainable": False, "description": "microsoft/DCVC CVPR2025；推理专用，需 setup"},
    {"id": "nevc", "name": "NEVC-1.0 (EHVC)", "family": "nevc", "kind": "learned-video", "ext": "bin", "is_neural": True, "qualities": [0, 1, 2, 3], "trainable": False, "description": "bytedance DCVC-derived；IntraNoAR+DMC+MLCodec_rans；HF checkpoint"},
    {"id": "dcvc_dc", "name": "DCVC-DC (CVPR2023)", "family": "dcvc_dc", "kind": "learned-video", "ext": "bin", "is_neural": True, "qualities": [0, 1, 2, 3], "trainable": False, "description": "microsoft DCVC family；IntraNoAR+DMC+MLCodec_rans；OneDrive checkpoint"},
    {"id": "lsmc", "name": "LSMC (Lossless SegMap)", "family": "lsmc", "kind": "learned-video", "ext": "bin", "is_neural": True, "qualities": [0], "trainable": False, "description": "InterDigital C++ CLI；链码+算术编码；无损单点 PSNR=inf"},
    {"id": "difftok", "name": "DiffTok VQ Tokenizer", "family": "difftok", "kind": "learned-video", "ext": "bin", "is_neural": True, "qualities": [1], "trainable": False, "description": "灰度轮廓 1D VQ tokenizer；BCE loss；TiTok 风格 latent tokens"},
    {"id": "img-bmshj2018-factorized", "name": "bmshj2018-factorized", "family": "img", "kind": "learned-image", "ext": "bin", "is_neural": True, "qualities": [1, 4, 8], "trainable": True, "description": "CompressAI 图像模型 bmshj2018-factorized（per-frame 当视频 codec）"},
    {"id": "img-bmshj2018-hyperprior", "name": "bmshj2018-hyperprior", "family": "img", "kind": "learned-image", "ext": "bin", "is_neural": True, "qualities": [1, 4, 8], "trainable": True, "description": "CompressAI 图像模型 bmshj2018-hyperprior"},
    {"id": "img-mbt2018-mean", "name": "mbt2018-mean", "family": "img", "kind": "learned-image", "ext": "bin", "is_neural": True, "qualities": [1, 4, 8], "trainable": True, "description": "CompressAI 图像模型 mbt2018-mean"},
    {"id": "img-mbt2018", "name": "mbt2018", "family": "img", "kind": "learned-image", "ext": "bin", "is_neural": True, "qualities": [1, 4, 8], "trainable": True, "description": "CompressAI 图像模型 mbt2018"},
    {"id": "img-cheng2020-anchor", "name": "cheng2020-anchor", "family": "img", "kind": "learned-image", "ext": "bin", "is_neural": True, "qualities": [1, 4, 6], "trainable": True, "description": "CompressAI 图像模型 cheng2020-anchor"},
    {"id": "img-cheng2020-attn", "name": "cheng2020-attn", "family": "img", "kind": "learned-image", "ext": "bin", "is_neural": True, "qualities": [1, 4, 6], "trainable": True, "description": "CompressAI 图像模型 cheng2020-attn"},
    {"id": "img-ELIC", "name": "ELIC", "family": "img", "kind": "learned-image", "ext": "bin", "is_neural": True, "qualities": [1, 4, 5], "trainable": True, "description": "CompressAI 图像模型 ELIC"},
]


def catalog() -> list[dict]:
    """Static codec catalog (no codec module import). Returns copies of _CODEC_CATALOG."""
    return [dict(c) for c in _CODEC_CATALOG]


def list_codecs() -> list[str]:
    """Static codec id list (no import of codec modules)."""
    return [c["id"] for c in _CODEC_CATALOG]
