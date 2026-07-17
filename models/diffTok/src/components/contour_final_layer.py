import torch
import torch.nn as nn
from .norm import RMSNorm


class ContourFinalLayer(nn.Module):
    """Output projection layer without any conditioning/modulation."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))
