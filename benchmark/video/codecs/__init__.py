"""Video codecs. Importing this package registers all built-in codecs."""

from .base import (  # noqa: F401
    CodecConfig,
    VideoCodec,
    CODEC_REGISTRY,
    build_codec,
    list_codecs,
    register_codec,
)

from . import x264 as _x264  # noqa: F401
from . import x265 as _x265  # noqa: F401
from . import svtav1 as _svtav1  # noqa: F401
from . import vp9 as _vp9  # noqa: F401
from . import mpeg4 as _mpeg4  # noqa: F401
from . import ssf2020 as _ssf2020  # noqa: F401
from . import dcvc_rt as _dcvc_rt  # noqa: F401
from . import learned_image as _learned_image  # noqa: F401

__all__ = [
    "CodecConfig",
    "VideoCodec",
    "CODEC_REGISTRY",
    "build_codec",
    "list_codecs",
    "register_codec",
    "catalog",
]


# ---- unified codec catalog (single source of truth for server + runner) ----

# qualities not derivable from the codec class for non-image codecs.
_NON_IMG_QUALITIES: dict[str, list[int]] = {
    "x264": [18, 23, 28, 33],
    "x265": [18, 23, 28, 33],
    "vp9": [18, 23, 28, 33],
    "mpeg4": [8, 14, 20, 26],
    "svtav1": [18, 23, 28, 33],
    "ssf2020": [1, 3, 5, 7, 9],
    "dcvc_rt": [20, 30, 40],
}

_CODEC_META: dict[str, tuple[str, str]] = {
    "x264": ("x264 (H.264/AVC)", "最通用基线 codec"),
    "x265": ("x265 (HEVC)", "高压缩比现代 codec"),
    "svtav1": ("SVT-AV1", "新一代 royalty-free，较慢（static-ffmpeg win32 无 libsvtav1）"),
    "vp9": ("VP9", "Google 开源 codec"),
    "mpeg4": ("MPEG-4 Part 2", "DivX/Xvid 系老一代基线；qscale 控质量，码率效率低于 H.264"),
    "ssf2020": ("ssf2020 (Scale-Space Flow)", "CompressAI 视频模型 CVPR2020；可 fine-tune"),
    "dcvc_rt": ("DCVC-RT (real-time NVC)", "microsoft/DCVC CVPR2025；推理专用，需 setup"),
}


def catalog() -> list[dict]:
    """Unified codec catalog. ``id`` matches ``CODEC_REGISTRY`` (so the server's
    codec selector and the runner speak the same ids — no more bmshj2018-* vs
    img-bmshj2018-* mismatch). Each entry:

        id, name, family, kind(=family), ext, is_neural, qualities, trainable, description

    ``kind`` is a clean category: ``"codec"`` (traditional ffmpeg, is_neural=False),
    ``"learned-video"`` (ssf2020 / dcvc_rt), ``"learned-image"`` (img-*).
    img-* qualities come from learned_image._IMG_MODELS; the rest from the tables
    above. Single source of truth: the server's /codecs endpoint and
    run_all_subprocess's CRF map both derive from this.
    """
    from .learned_image import _IMG_MODELS
    imgq = {name: quals for name, quals in _IMG_MODELS}
    out: list[dict] = []
    for cid in list_codecs():
        cls = CODEC_REGISTRY[cid]
        family = getattr(cls, "family", "codec")
        ext = getattr(cls, "ext", "mp4")
        is_neural = bool(getattr(cls, "is_neural", False))
        trainable = False
        if cid.startswith("img-"):
            base = cid[len("img-"):]
            quals = list(imgq.get(base, []))
            name = base
            desc = f"CompressAI 图像模型 {base}（per-frame 当视频 codec）"
            trainable = True
            kind = "learned-image"
        elif cid == "ssf2020":
            quals = _NON_IMG_QUALITIES["ssf2020"]
            name, desc = _CODEC_META["ssf2020"]
            trainable = True
            kind = "learned-video"
        elif cid == "dcvc_rt":
            quals = _NON_IMG_QUALITIES["dcvc_rt"]
            name, desc = _CODEC_META["dcvc_rt"]
            trainable = False
            kind = "learned-video"
        else:
            quals = _NON_IMG_QUALITIES.get(cid, [18, 23, 28, 33])
            name, desc = _CODEC_META.get(cid, (cid, ""))
            kind = "codec"
        out.append({
            "id": cid, "name": name, "family": family, "kind": kind,
            "ext": ext, "is_neural": is_neural,
            "qualities": quals, "trainable": trainable, "description": desc,
        })
    return out
