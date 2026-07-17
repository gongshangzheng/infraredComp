import math
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from ..components.contour_block import ContourBlock
from ..components.embed import BottleneckPatchEmbed
from ..components.norm import RMSNorm


class ContourEncoder(nn.Module):
    """
    ViT-style encoder for grayscale contour images.

    Concatenates image patches with learnable latent tokens, runs them
    through transformer blocks, and returns only the latent token outputs
    as the compressed representation.

    Token sequence: [image_patches | latent_tokens]
    Output: latent_tokens [B, num_latent, dim]
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        in_chans: int = 1,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_latent: int = 64,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.num_latent = num_latent
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        # Patch embedding: Conv2d bottleneck
        self.patch_embed = BottleneckPatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            pca_dim=dim,
            embed_dim=dim,
        )

        # Learnable positional embedding for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))

        # Learnable latent tokens (extra tokens that collect the compressed info)
        self.latent_tokens = nn.Parameter(torch.zeros(1, num_latent, dim))

        # Learnable positional embedding for latent tokens
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, num_latent, dim))

        self.blocks = nn.ModuleList([
            ContourBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                num_latents=num_latent,
                qk_norm=qk_norm,
            )
            for _ in range(depth)
        ])
        self.ln_post = RMSNorm(dim, eps=1e-6)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.latent_tokens, std=0.02)
        trunc_normal_(self.latent_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input image (C=1 for grayscale)
        Returns:
            latent_tokens: [B, num_latent, dim]
        """
        B = x.shape[0]

        # Embed patches
        patches = self.patch_embed(x)              # [B, num_patches, dim]
        patches = patches + self.pos_embed         # add positional encoding

        # Expand learnable latent tokens for the batch
        latents = self.latent_tokens.expand(B, -1, -1)
        latents = latents + self.latent_pos_embed  # add latent positional encoding

        # Concatenate: [patches | latents]
        tokens = torch.cat([patches, latents], dim=1)  # [B, num_patches+num_latent, dim]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.ln_post(tokens)

        # Extract latent tokens (last num_latent positions)
        return tokens[:, self.num_patches:]        # [B, num_latent, dim]
