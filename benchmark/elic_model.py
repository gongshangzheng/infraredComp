"""ELIC (Efficient Learned Image Compression) model adapter for CompressAI 1.2.8.

Based on the reimplementation by VincentChandelier:
https://github.com/VincentChandelier/ELiC-ReImplemetation

Original paper: "ELIC: Efficient Learned Image Compression with Unevenly Grouped
Space-Channel Contextual Adaptive Coding" (CVPR 2022).

This module provides a self-contained ELIC model class that can load pretrained
checkpoints from VincentChandelier's repo and expose the same compress/decompress
interface used by CompressAI models.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.ops import quantize_ste
from compressai.ops.parametrizers import NonNegativeParametrizer

# ---------------------------------------------------------------------------
# Checkpoint registry: lambda -> (filename, Google Drive ID)
# ---------------------------------------------------------------------------
ELIC_CHECKPOINTS = {
    1: ("ELIC_lambda004.pth.tar", "1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q"),  # ~0.1 bpp
    2: ("ELIC_lambda008.pth.tar", "1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL"),  # ~0.2 bpp
    3: ("ELIC_lambda016.pth.tar", "1MWlYAmpHbWlGtG7MBBTPEew800grY5yC"),  # ~0.35 bpp
    4: ("ELIC_lambda032.pth.tar", "1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc"),  # ~0.5 bpp
    5: ("ELIC_lambda150.pth.tar", "1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL"),  # ~1.0 bpp
    6: ("ELIC_lambda450.pth.tar", "1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN"),  # ~1.5 bpp
}

ELIC_CKPT_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "elic"

# Quality levels we actually use in the benchmark
ELIC_QUALITIES = [1, 4, 5]  # low, medium, high bitrate

# ---------------------------------------------------------------------------
# Layer definitions (from VincentChandelier's ELICUtilis)
# ---------------------------------------------------------------------------

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


class GDN(nn.Module):
    """Generalized Divisive Normalization layer."""

    def __init__(self, in_channels: int, inverse: bool = False,
                 beta_min: float = 1e-6, gamma_init: float = 0.1):
        super().__init__()
        self.inverse = bool(inverse)
        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)
        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)
        norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)
        return x * norm


class ResidualBottleneckBlock(nn.Module):
    """Residual block with two 3x3 convolutions (bottleneck)."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch // 2, in_ch // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch // 2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out + identity


class AttentionBlock(nn.Module):
    """Self-attention block from Cheng2020."""

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2), nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2), nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                out = self.conv(x) + x
                return self.relu(out)

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())
        self.conv_b = nn.Sequential(
            ResidualUnit(), ResidualUnit(), ResidualUnit(), conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_a(x) * torch.sigmoid(self.conv_b(x)) + x


class CheckboardMaskedConv2d(nn.Conv2d):
    """Checkerboard-masked convolution for anchor/non-anchor split."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.zeros_like(self.weight.data))
        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class Quantizer:
    """Simple quantizer with noise/STE modes."""

    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            noise = torch.empty_like(inputs).uniform_(-0.5, 0.5)
            return inputs + noise
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)


# ---------------------------------------------------------------------------
# ELIC Model
# ---------------------------------------------------------------------------

class ELICModel(CompressionModel):
    """ELIC: Efficient Learned Image Compression (CVPR 2022).

    Unevenly grouped space-channel contextual adaptive coding.
    Based on VincentChandelier's reimplementation, adapted for CompressAI 1.2.8.
    """

    def __init__(self, N=192, M=320, num_slices=5, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)
        self.num_slices = num_slices
        self.groups = [0, 16, 16, 32, 64, 192]

        # Analysis transform
        self.g_a = nn.Sequential(
            conv(3, N),
            *[ResidualBottleneckBlock(N) for _ in range(3)],
            conv(N, N),
            *[ResidualBottleneckBlock(N) for _ in range(3)],
            AttentionBlock(N),
            conv(N, N),
            *[ResidualBottleneckBlock(N) for _ in range(3)],
            conv(N, M),
            AttentionBlock(M),
        )

        # Synthesis transform
        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N),
            *[ResidualBottleneckBlock(N) for _ in range(3)],
            deconv(N, N),
            AttentionBlock(N),
            *[ResidualBottleneckBlock(N) for _ in range(3)],
            deconv(N, N),
            *[ResidualBottleneckBlock(N) for _ in range(3)],
            deconv(N, 3),
        )

        # Hyper transforms
        self.h_a = nn.Sequential(
            conv3x3(M, N), nn.ReLU(inplace=True),
            conv(N, N), nn.ReLU(inplace=True),
            conv(N, N),
        )
        self.h_s = nn.Sequential(
            deconv(N, N), nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2), nn.ReLU(inplace=True),
            conv3x3(N * 3 // 2, 2 * M),
        )

        # Channel context transforms
        self.cc_transforms = nn.ModuleList([
            nn.Sequential(
                conv(self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0],
                     224, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[i + 1] * 2, stride=1, kernel_size=5),
            ) for i in range(1, num_slices)
        ])

        # Spatial context (checkerboard)
        self.context_prediction = nn.ModuleList([
            CheckboardMaskedConv2d(
                self.groups[i + 1], 2 * self.groups[i + 1],
                kernel_size=5, padding=2, stride=1,
            ) for i in range(num_slices)
        ])

        # Parameter aggregation
        self.ParamAggregation = nn.ModuleList([
            nn.Sequential(
                conv1x1(640 + self.groups[i + 1 if i > 0 else 0] * 2
                        + self.groups[i + 1] * 2, 640),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[i + 1] * 2),
            ) for i in range(num_slices)
        ])

        self.quantizer = Quantizer()
        self.gaussian_conditional = GaussianConditional(None)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, noisequant=False):
        y = self.g_a(x)
        B, C, H, W = y.size()

        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = torch.round(z - z_offset) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        anchor = torch.zeros_like(y)
        non_anchor = torch.zeros_like(y)
        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)
        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W).to(x.device),
            [2 * i for i in self.groups[1:]], 1,
        )
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)
            else:
                support_slices = torch.cat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.chunk(2, 1)

            support = (torch.cat([latent_means, latent_scales], dim=1) if slice_index == 0
                       else torch.cat([support_slices_ch_mean, support_slices_ch_scale,
                                       latent_means, latent_scales], dim=1))

            # Checkerboard process 1 (anchor)
            y_anchor = anchor_split[slice_index]
            means_anchor, scales_anchor = self.ParamAggregation[slice_index](
                torch.cat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor)
            means_hat_split = torch.zeros_like(y_anchor)
            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            if noisequant:
                y_anchor_q = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_q_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_q = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor
                y_anchor_q_gs = self.quantizer.quantize(y_anchor - means_anchor, "ste") + means_anchor

            y_anchor_q[:, :, 0::2, 1::2] = 0
            y_anchor_q[:, :, 1::2, 0::2] = 0
            y_anchor_q_gs[:, :, 0::2, 1::2] = 0
            y_anchor_q_gs[:, :, 1::2, 0::2] = 0

            # Checkerboard process 2 (non-anchor)
            masked_context = self.context_prediction[slice_index](y_anchor_q)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]

            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scales_hat_split, means=means_hat_split,
            )

            y_non_anchor = non_anchor_split[slice_index]
            if noisequant:
                y_na_q = self.quantizer.quantize(y_non_anchor, "noise")
                y_na_q_gs = self.quantizer.quantize(y_non_anchor, "ste")
            else:
                y_na_q = self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste") + means_non_anchor
                y_na_q_gs = self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste") + means_non_anchor

            y_na_q[:, :, 0::2, 0::2] = 0
            y_na_q[:, :, 1::2, 1::2] = 0
            y_na_q_gs[:, :, 0::2, 0::2] = 0
            y_na_q_gs[:, :, 1::2, 1::2] = 0

            y_hat_slices.append(y_anchor_q + y_na_q)
            y_hat_slices_for_gs.append(y_anchor_q_gs + y_na_q_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        """Compress input tensor to bitstrings."""
        y = self.g_a(x)
        B, C, H, W = y.size()

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        y_slices = torch.split(y, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W).to(x.device),
            [2 * i for i in self.groups[1:]], 1,
        )

        y_strings = []
        y_hat_slices = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                ch = self.cc_transforms[slice_index - 1](support_slices)
                ch_mean, ch_scale = ch.chunk(2, 1)
            else:
                support_slices = torch.cat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                ch = self.cc_transforms[slice_index - 1](support_slices)
                ch_mean, ch_scale = ch.chunk(2, 1)

            support = (torch.cat([latent_means, latent_scales], dim=1) if slice_index == 0
                       else torch.cat([ch_mean, ch_scale, latent_means, latent_scales], dim=1))

            # Anchor
            y_anchor = y_slices[slice_index].clone()
            means_anchor, scales_anchor = self.ParamAggregation[slice_index](
                torch.cat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            Ba, Ca, Ha, Wa = y_anchor.size()
            y_a_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(x.device)
            m_a_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(x.device)
            s_a_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(x.device)
            y_a_dec = torch.zeros(Ba, Ca, Ha, Wa).to(x.device)

            y_a_enc[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_a_enc[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            m_a_enc[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            m_a_enc[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            s_a_enc[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            s_a_enc[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            idx_a = self.gaussian_conditional.build_indexes(s_a_enc)
            a_strings = self.gaussian_conditional.compress(y_a_enc, idx_a, means=m_a_enc)
            a_quant = self.gaussian_conditional.decompress(a_strings, idx_a, means=m_a_enc)
            y_a_dec[:, :, 0::2, 0::2] = a_quant[:, :, 0::2, :]
            y_a_dec[:, :, 1::2, 1::2] = a_quant[:, :, 1::2, :]

            # Non-anchor
            masked_ctx = self.context_prediction[slice_index](y_a_dec)
            m_na, s_na = self.ParamAggregation[slice_index](
                torch.cat([masked_ctx, support], dim=1)
            ).chunk(2, 1)

            y_na_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(x.device)
            m_na_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(x.device)
            s_na_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(x.device)

            non_anchor = y_slices[slice_index].clone()
            y_na_enc[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_na_enc[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            m_na_enc[:, :, 0::2, :] = m_na[:, :, 0::2, 1::2]
            m_na_enc[:, :, 1::2, :] = m_na[:, :, 1::2, 0::2]
            s_na_enc[:, :, 0::2, :] = s_na[:, :, 0::2, 1::2]
            s_na_enc[:, :, 1::2, :] = s_na[:, :, 1::2, 0::2]

            idx_na = self.gaussian_conditional.build_indexes(s_na_enc)
            na_strings = self.gaussian_conditional.compress(y_na_enc, idx_na, means=m_na_enc)
            na_quant = self.gaussian_conditional.decompress(na_strings, idx_na, means=m_na_enc)

            y_na_q = torch.zeros_like(means_anchor)
            y_na_q[:, :, 0::2, 1::2] = na_quant[:, :, 0::2, :]
            y_na_q[:, :, 1::2, 0::2] = na_quant[:, :, 1::2, :]

            y_hat_slices.append(y_a_dec + y_na_q)
            y_strings.append([a_strings, na_strings])

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        """Decompress bitstrings to tensor."""
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B = z_hat.size(0)
        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        y_strings = strings[0]
        ctx_params_anchor = torch.zeros(
            (B, self.M * 2, z_hat.shape[2] * 4, z_hat.shape[3] * 4),
        ).to(z_hat.device)
        ctx_params_anchor_split = torch.split(
            ctx_params_anchor, [2 * i for i in self.groups[1:]], 1,
        )

        y_hat_slices = []
        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                ch = self.cc_transforms[slice_index - 1](support_slices)
                ch_mean, ch_scale = ch.chunk(2, 1)
            else:
                support_slices = torch.cat([y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1)
                ch = self.cc_transforms[slice_index - 1](support_slices)
                ch_mean, ch_scale = ch.chunk(2, 1)

            support = (torch.cat([latent_means, latent_scales], dim=1) if slice_index == 0
                       else torch.cat([ch_mean, ch_scale, latent_means, latent_scales], dim=1))

            m_anchor, s_anchor = self.ParamAggregation[slice_index](
                torch.cat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            Ba, Ca, Ha, Wa = m_anchor.size()
            m_a_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(z_hat.device)
            s_a_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(z_hat.device)
            y_a_dec = torch.zeros(Ba, Ca, Ha, Wa).to(z_hat.device)

            m_a_enc[:, :, 0::2, :] = m_anchor[:, :, 0::2, 0::2]
            m_a_enc[:, :, 1::2, :] = m_anchor[:, :, 1::2, 1::2]
            s_a_enc[:, :, 0::2, :] = s_anchor[:, :, 0::2, 0::2]
            s_a_enc[:, :, 1::2, :] = s_anchor[:, :, 1::2, 1::2]

            idx_a = self.gaussian_conditional.build_indexes(s_a_enc)
            a_quant = self.gaussian_conditional.decompress(
                y_strings[slice_index][0], idx_a, means=m_a_enc,
            )
            y_a_dec[:, :, 0::2, 0::2] = a_quant[:, :, 0::2, :]
            y_a_dec[:, :, 1::2, 1::2] = a_quant[:, :, 1::2, :]

            masked_ctx = self.context_prediction[slice_index](y_a_dec)
            m_na, s_na = self.ParamAggregation[slice_index](
                torch.cat([masked_ctx, support], dim=1)
            ).chunk(2, 1)

            m_na_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(z_hat.device)
            s_na_enc = torch.zeros(Ba, Ca, Ha, Wa // 2).to(z_hat.device)
            m_na_enc[:, :, 0::2, :] = m_na[:, :, 0::2, 1::2]
            m_na_enc[:, :, 1::2, :] = m_na[:, :, 1::2, 0::2]
            s_na_enc[:, :, 0::2, :] = s_na[:, :, 0::2, 1::2]
            s_na_enc[:, :, 1::2, :] = s_na[:, :, 1::2, 0::2]

            idx_na = self.gaussian_conditional.build_indexes(s_na_enc)
            na_quant = self.gaussian_conditional.decompress(
                y_strings[slice_index][1], idx_na, means=m_na_enc,
            )

            y_na_q = torch.zeros_like(m_anchor)
            y_na_q[:, :, 0::2, 1::2] = na_quant[:, :, 0::2, :]
            y_na_q[:, :, 1::2, 0::2] = na_quant[:, :, 1::2, :]

            y_hat_slices.append(y_a_dec + y_na_q)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_elic_model(quality: int, device: str = "cpu", checkpoint_path: str | None = None) -> ELICModel:
    """Load an ELIC model for the given quality level.

    If ``checkpoint_path`` is given, load that trained state_dict instead of the
    pretrained Google-Drive checkpoint (checkpoint→eval hook: use a model trained
    via scripts/train_model.py). Otherwise load the pretrained checkpoint.

    Returns
    -------
    ELICModel in eval mode with updated entropy model.
    """
    if checkpoint_path:
        from pathlib import Path
        cp = Path(checkpoint_path)
        if not cp.is_file():
            raise RuntimeError(f"Trained ELIC checkpoint not found: {cp}")
        state_dict = torch.load(cp, map_location="cpu", weights_only=False)
    else:
        if quality not in ELIC_CHECKPOINTS:
            raise ValueError(f"ELIC quality {quality} not in registry. "
                             f"Available: {sorted(ELIC_CHECKPOINTS.keys())}")
        fname, _ = ELIC_CHECKPOINTS[quality]
        ckpt_path = ELIC_CKPT_DIR / fname
        if not ckpt_path.is_file():
            raise RuntimeError(
                f"ELIC checkpoint not found: {ckpt_path}\n"
                f"Download from Google Drive ID: {ELIC_CHECKPOINTS[quality][1]}"
            )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = ELICModel(N=192, M=320, num_slices=5)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    model.update()
    return model
