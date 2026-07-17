import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.init import trunc_normal_

from ..components.contour_block import ContourBlock
from ..components.contour_final_layer import ContourFinalLayer


class ContourDecoder(nn.Module):
    """
    ViT-style decoder for grayscale contour images.

    Takes quantized latent tokens as input, uses learnable mask tokens as
    image patch slots, runs them through transformer blocks, and projects
    back to pixel space.

    Token sequence: [mask_tokens (num_patches) | latent_tokens (num_latent)]
    Output: [B, out_chans, H, W] logits (before sigmoid)
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 8,
        out_chans: int = 1,
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
        self.patch_size = patch_size
        self.out_chans = out_chans
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches
        h = w = image_size // patch_size
        self.h = h
        self.w = w

        # Learnable mask tokens (image patch slots — no real input image)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
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

        self.final_layer = ContourFinalLayer(
            hidden_size=dim,
            patch_size=patch_size,
            out_channels=out_chans,
        )

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
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

    def forward(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_tokens: [B, num_latent, dim] quantized latent tokens
        Returns:
            logits: [B, out_chans, H, W] (before sigmoid)
        """
        B = latent_tokens.shape[0]

        # Expand mask tokens to fill all patch slots
        image_tokens = self.mask_token.expand(B, self.num_patches, -1)
        image_tokens = image_tokens + self.pos_embed  # add positional encoding

        # Add positional encoding to latent tokens
        latents = latent_tokens + self.latent_pos_embed

        # Concatenate: [mask_tokens | latent_tokens]
        tokens = torch.cat([image_tokens, latents], dim=1)  # [B, num_patches+num_latent, dim]

        for block in self.blocks:
            tokens = block(tokens)

        # Extract patch tokens (first num_patches positions)
        patch_tokens = tokens[:, :self.num_patches]  # [B, num_patches, dim]

        # Project to pixel space
        patch_tokens = self.final_layer(patch_tokens)  # [B, num_patches, patch_size²*out_chans]

        # Unpatchify: [B, num_patches, patch_size²*C] → [B, C, H, W]
        x = rearrange(
            patch_tokens,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=self.h, w=self.w,
            p1=self.patch_size, p2=self.patch_size,
            c=self.out_chans,
        )
        return x
