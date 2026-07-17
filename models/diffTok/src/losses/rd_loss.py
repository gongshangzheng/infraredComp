"""RD loss for diffTok contour VQ autoencoder."""
import math
import torch
import torch.nn.functional as F


def rd_loss_difftok(logits, x0, vq_loss, pos_weight: float = 10.0) -> tuple:
    """BCE loss for binary contour maps + VQ commitment loss.

    pos_weight upweights edge pixels to compensate class imbalance (~5-15% edges).
    Returns (loss, loss_val, psnr_proxy, bpp_dummy) matching rd_loss tuple shape.
    """
    pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, x0, pos_weight=pw)
    loss = bce + vq_loss
    with torch.no_grad():
        x_hat = torch.sigmoid(logits)
        mse = torch.mean((x_hat - x0) ** 2).item()
        psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
    return loss, loss.item(), psnr, 0.0
