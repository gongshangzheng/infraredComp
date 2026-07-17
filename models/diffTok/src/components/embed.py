"""
Embedding modules for PoET.

Provides simple linear projection embeddings for token and feature inputs.
"""

import torch.nn as nn

class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding with bottleneck architecture.

    Input: [B, C, H, W]
    Output: [B, num_patches, embed_dim]

    Uses two-stage projection:
    1. proj1: Conv2d for patchification + PCA bottleneck
    2. proj2: Conv2d for projection to embed_dim

    Args:
        img_size: Input image size (default: 224)
        patch_size: Patch size (default: 16)
        in_chans: Number of input channels (default: 3)
        pca_dim: Intermediate bottleneck dimension (default: 768)
        embed_dim: Output embedding dimension (default: 768)
        bias: Whether to use bias in proj2 (default: True)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class Embed(nn.Module):
    """
    Simple linear embedding layer with optional normalization.

    This module projects input features to an embedding dimension,
    with optional layer normalization.

    Args:
        in_chans: Number of input channels
        embed_dim: Output embedding dimension
        norm_layer: Normalization layer class (optional)
        bias: Whether to use bias in linear projection

    Forward:
        Input: (batch, seq_len, in_chans)
        Output: (batch, seq_len, embed_dim)
    """
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        norm_layer=None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Forward pass of Embed layer.

        Args:
            x: Input tensor of shape (batch, seq_len, in_chans)

        Returns:
            Embedded tensor of shape (batch, seq_len, embed_dim)
        """
        x = self.proj(x)
        x = self.norm(x)
        return x
