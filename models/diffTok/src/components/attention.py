import torch
import torch.nn as nn
from functools import partial

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None

from .rope import VisionRotaryEmbedding, SequentialRotaryEmbedding
from .norm import RMSNorm
import torch.nn.functional as F

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., flash=True,
                 rope_size=0, rope_reg_size=0, rope_latent_size=0, num_registers=0, num_latents=0, reg_theta=10000, latent_theta=1000, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Use Identity instead of Dropout when attn_drop=0 to avoid creating unnecessary modules
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        # Use Identity instead of Dropout when proj_drop=0 to avoid creating unnecessary modules
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.flash = flash
        self.num_registers = num_registers
        # num_latents stands for the number of latent tokens, while latent_rope_size stands for the number of the latent tokens that will be rotated
        self.num_latents = num_latents

        # 2D RoPE needs head_dim//2, because it has x and y two axes, each using half of the head_dim
        self.rope = VisionRotaryEmbedding(head_dim//2, rope_size) if rope_size > 0 else None  # 2D RoPE for patches
        self.rope_reg = VisionRotaryEmbedding(head_dim//2, rope_reg_size, theta=reg_theta) if rope_reg_size > 0 else None  # 2D RoPE for registers
        # 1D RoPE needs head_dim, because it has only one axis, using the whole head_dim
        if self.num_latents > 0:
            self.latent_rope = SequentialRotaryEmbedding(head_dim, rope_latent_size, theta=latent_theta) if rope_latent_size > 0 else None  # 1D RoPE for latents
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, attn_mask=None, is_causal=False):
        if attn_mask is not None and is_causal:
            raise ValueError("attn_mask and is_causal should not be used together")
        B, N, C = x.shape
        reg_idx = N - self.num_registers - self.num_latents
        latent_idx = N - self.num_latents

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        # class | patches | registers | latents
        if self.rope is not None:
            q[:, 1:reg_idx] = self.rope(q[:, 1:reg_idx])
            k[:, 1:reg_idx] = self.rope(k[:, 1:reg_idx])

        if self.rope_reg is not None:
            q[:, reg_idx:latent_idx] = self.rope_reg(q[:, reg_idx:latent_idx])
            k[:, reg_idx:latent_idx] = self.rope_reg(k[:, reg_idx:latent_idx])

        if self.latent_rope is not None:
            q[:, latent_idx:] = self.latent_rope(q[:, latent_idx:])
            k[:, latent_idx:] = self.latent_rope(k[:, latent_idx:])

        # FlashAttention path
        if self.flash and attn_mask is None:
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(
                qkv,
                causal=is_causal
            ).reshape(B, N, C)

        else:
            # SDPA fallback
            q = q.transpose(1, 2)  # B H N D
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Expand attn_mask to include head dimension if provided
            # Input: [B, N, N] -> Output: [B, 1, N, N] for broadcasting across heads
            if attn_mask is not None and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)

            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if (self.training and isinstance(self.attn_drop, nn.Dropout)) else 0.0,
                is_causal=is_causal
            )

            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x