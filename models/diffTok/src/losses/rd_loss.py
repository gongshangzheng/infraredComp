"""RD loss for diffTok contour VQ autoencoder."""
import math
import torch
import torch.nn.functional as F


def rd_loss_difftok(logits, x0, vq_loss, indices, pos_weight: float = 10.0) -> tuple:
    """BCE loss for binary contour maps + VQ commitment loss.

    pos_weight upweights edge pixels to compensate class imbalance (~5-15% edges).
    bpp is estimated from VQ token indices via empirical marginal entropy:
      H = -Σ p_i·log2(p_i) over codebook usage in the batch,
      bpp = H · num_latent / num_pixels.
    Returns (loss, loss_val, psnr_proxy, bpp).
    """
    pw = torch.tensor([pos_weight], dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, x0, pos_weight=pw)
    loss = bce + vq_loss
    with torch.no_grad():
        x_hat = torch.sigmoid(logits)
        mse = torch.mean((x_hat - x0) ** 2).item()
        psnr = 10.0 * math.log10(1.0 / (mse + 1e-10))
        num_pixels = x0.shape[-1] * x0.shape[-2] * x0.shape[-3]
        num_latent = indices.shape[1]
        hist = torch.bincount(indices.flatten()).float()
        p = hist / hist.sum()
        p = p[p > 0]
        entropy = -(p * torch.log2(p)).sum().item()
        bpp = entropy * num_latent / num_pixels
    return loss, loss.item(), psnr, bpp
