"""MPEG-4 Part 2 (Advanced Simple Profile) via libavcodec ``mpeg4`` — DivX/Xvid 系基线。

与已接入的 x264 (MPEG-4 **Part 10** / AVC) 是不同标准：Part 2 是更早的、
码率效率明显低于 H.264 的基线，常作为"老一代"对照组。

注意：``mpeg4`` encoder 不支持 ``-crf``（CRF 是 x264/x265/vpx/av1 的码率控制）。
Part 2 用 ``-qscale:v``（1-31，越小质量越好、码率越高，**方向与 CRF 一致**）。
框架的 quality 整数（结果里仍写作 ``crf<N>``，与 ssf2020 复用 crf 槽位同理）
在此被当作 qscale 传入。
"""

from .base import VideoCodec, register_codec


@register_codec("mpeg4")
class Mpeg4Codec(VideoCodec):
    name = "mpeg4"
    family = "mpeg4"
    encoder = "mpeg4"          # libavcodec 原生 MPEG-4 Part 2 encoder
    default_preset = None      # mpeg4 无 preset 概念
    ext = "mp4"
    # 浏览器不解码 MPEG-4 Part 2 → harness 从重建帧合成 H.264 可播 mp4 覆盖展示路径
    # (码流大小已在编码后测得, bpp/bitrate 仍反映 Part 2 压缩, 不受影响)。
    browser_playable = False

    def encode_args(self, frames_dir, fps, bitstream):  # type: ignore[override]
        # base 生成 "... -c:v mpeg4 -crf <q> -pix_fmt yuv420p <bitstream>"；
        # mpeg4 不支持 -crf，把 "-crf", "<q>" 替换为 "-qscale:v", "<q>"。
        args = super().encode_args(frames_dir, fps, bitstream)
        out: list[str] = []
        i = 0
        while i < len(args):
            if args[i] == "-crf":
                out += ["-qscale:v", args[i + 1]]
                i += 2
            else:
                out.append(args[i])
                i += 1
        return out
