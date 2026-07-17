import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

from ..encoders.contour_encoder import ContourEncoder
from ..decoders.contour_decoder import ContourDecoder


class ContourVQAE(nn.Module):
    """
    Contour VQ Autoencoder (TiTok-style 1D tokenizer for grayscale edge maps).

    Architecture: ContourEncoder → Linear → VQ → Linear → ContourDecoder

    Forward returns (logits, commit_loss, indices).
    Use encode_indices() / decode_indices() for inference.
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        q = cfg.quantizer

        self.encoder = ContourEncoder(
            image_size=m.image_size,
            patch_size=m.patch_size,
            in_chans=m.inout_chans,
            dim=m.encoder.dim,
            depth=m.encoder.depth,
            num_heads=m.encoder.num_heads,
            mlp_ratio=m.get("mlp_ratio", 4.0),
            num_latent=m.num_latent,
            qkv_bias=m.get("qkv_bias", True),
            drop=m.get("drop", 0.0),
            attn_drop=m.get("attn_drop", 0.0),
            qk_norm=m.get("qk_norm", False),
        )

        self.decoder = ContourDecoder(
            image_size=m.image_size,
            patch_size=m.patch_size,
            out_chans=m.inout_chans,
            dim=m.decoder.dim,
            depth=m.decoder.depth,
            num_heads=m.decoder.num_heads,
            mlp_ratio=m.get("mlp_ratio", 4.0),
            num_latent=m.num_latent,
            qkv_bias=m.get("qkv_bias", True),
            drop=m.get("drop", 0.0),
            attn_drop=m.get("attn_drop", 0.0),
            qk_norm=m.get("qk_norm", False),
        )

        self.quantizer = VectorQuantize(
            dim=q.token_dim,
            codebook_size=q.codebook_size,
            decay=q.get("decay", 0.99),
            commitment_weight=q.get("commitment_cost", 0.25),
            threshold_ema_dead_code=q.get("dead_code_threshold", 1.0),
            use_cosine_sim=q.get("use_l2_norm", False),
        )

        enc_dim = m.encoder.dim
        dec_dim = m.decoder.dim
        token_dim = q.token_dim
        self.pre_quant = nn.Linear(enc_dim, token_dim)
        self.post_quant = nn.Linear(token_dim, dec_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W] input image (C=1 for grayscale), values in [0, 1]
        Returns:
            (logits [B, C, H, W], commit_loss scalar, indices [B, num_latent])
        """
        z = self.encoder(x)                              # [B, num_latent, enc_dim]
        z = self.pre_quant(z)                            # [B, num_latent, token_dim]
        z_q, indices, commit_loss = self.quantizer(z)   # [B, num_latent, token_dim]
        z_q = self.post_quant(z_q)                       # [B, num_latent, dec_dim]
        logits = self.decoder(z_q)                       # [B, C, H, W]
        return logits, commit_loss, indices

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            indices: [B, num_latent] integer token ids
        """
        z = self.encoder(x)
        z = self.pre_quant(z)
        _, indices, _ = self.quantizer(z)
        return indices  # [B, num_latent]

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            image: [B, C, H, W] reconstructed image, values in [0, 1]
        """
        z_q = self.quantizer.get_output_from_indices(indices)  # [B, num_latent, token_dim]
        z_q = self.post_quant(z_q)                             # [B, num_latent, dec_dim]
        logits = self.decoder(z_q)                             # [B, C, H, W]
        return torch.sigmoid(logits)
