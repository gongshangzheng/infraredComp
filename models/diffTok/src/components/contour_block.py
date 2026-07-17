import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention
from .norm import RMSNorm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class ContourBlock(nn.Module):
    """Pre-norm ViT block without any conditioning/modulation."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        flash: bool = False,
        num_latents: int = 0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            flash=flash,
            num_latents=num_latents,
            qk_norm=qk_norm,
        )
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x
