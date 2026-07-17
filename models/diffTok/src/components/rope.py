from math import pi
import torch.nn.functional as F
import math

import torch
from torch import nn

from einops import rearrange, repeat
import numpy as np

def broadcat(freqss, dim = -1):
    num_freqss = len(freqss)
    shape_lens = set(list(map(lambda t: len(t.shape), freqss)))
    assert len(shape_lens) == 1, 'freqss must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), freqss)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_freqss), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    freqss = list(map(lambda t: t[0].expand(*t[1]), zip(freqss, expandable_shapes)))
    return torch.cat(freqss, dim = dim)

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

def rotate_freqs(freqs, angle_deg):
    assert freqs.ndim == 4 and freqs.shape[0] == freqs.shape[1], "Input must have shape (n, n, d1, d2)"
    n, _, d1, d2 = freqs.shape
    freq_type = freqs.dtype
    angle_rad = math.radians(angle_deg)

    # Reshape from (n, n, d1, d2) → (n, n, d1 * d2)
    freqs = freqs.reshape(n, n, -1)

    # Permute to (1, C, H, W) where C = d1 * d2
    freqs = freqs.permute(2, 0, 1).unsqueeze(0)

    # Rotation matrix (2x3)
    theta = torch.tensor([
        [ math.cos(angle_rad), -math.sin(angle_rad), 0.0],
        [ math.sin(angle_rad),  math.cos(angle_rad), 0.0]
    ], dtype=torch.float32, device=freqs.device).unsqueeze(0)

    freqs = freqs.to(torch.float32)

    # Build sampling grid
    grid = F.affine_grid(theta, freqs.size(), align_corners=True)

    # Rotate using bilinear interpolation, with border padding
    rotated = F.grid_sample(freqs, grid, mode='bilinear', padding_mode='border', align_corners=True)

    # Convert back: (1, C, H, W) → (H, W, C)
    rotated = rotated.squeeze(0).permute(1, 2, 0).to(freq_type)

    # Reshape back to (n, n, d1, d2)
    return rotated.reshape(n, n, d1, d2)

class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=14,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.pt_seq_len=pt_seq_len #  pt means pre-training
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        ft_seq_len = int(np.sqrt(x.shape[1])) # ft means fine-tuning
        t = torch.arange(ft_seq_len, device=x.device) / ft_seq_len * self.pt_seq_len # 归一化以后的位置坐标，用于处理动态长度

        freqs = torch.einsum('..., f -> ... f', t, self.freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2) # 14*32  # for cos and sin
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1) # 14*14*64  # for x and y
        # 每一个 tokens 在这个 head 上假设有 64 个维度，其中 32 个用于 x，32 个用于 y

        freqs_cos = freqs.cos().view(-1, 1, freqs.shape[-1]) # 计算余弦值
        freqs_sin = freqs.sin().view(-1, 1, freqs.shape[-1]) # 计算正弦值
        return  x * freqs_cos + rotate_half(x) * freqs_sin

class SequentialRotaryEmbedding(nn.Module):
    """1D Rotary Position Embedding for sequence tokens"""
    def __init__(self, dim, pt_seq_len=None, theta=1000):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len  # pt means pre-training
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        x: (batch, seq_len, num_heads, head_dim)
        """
        assert x.shape[-1] == self.dim
        seq_len = x.shape[1]
        device = x.device

        # 如果设置了pt_seq_len，将当前序列长度缩放到预训练尺度
        if self.pt_seq_len is not None:
            t = torch.arange(seq_len, device=device).float() / seq_len * self.pt_seq_len
        else:
            t = torch.arange(seq_len, device=device).float()

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        cos = freqs.cos()[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = freqs.sin()[None, :, None, :]  # (1, seq_len, 1, dim)

        return x * cos + rotate_half(x) * sin