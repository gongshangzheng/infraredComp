import torch
import torch.nn as nn

from ..encoders.contour_encoder import ContourEncoder
from ..decoders.contour_decoder import ContourDecoder
from ..quantizers.quantizer1d import VectorQuantizer1d


class ContourVQAE(nn.Module):
    """
    Contour VQ Autoencoder (TiTok-style 1D tokenizer for grayscale edge maps).

    Architecture: ContourEncoder → Linear → VQ → Linear → ContourDecoder

    Forward returns (logits, vq_result_dict) where logits are pre-sigmoid pixel values.
    Use encode_indices() / decode_indices() for inference.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: OmegaConf config with model.* and quantizer.* fields
        """
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

        self.quantizer = VectorQuantizer1d(
            codebook_size=q.codebook_size,
            token_dim=q.token_dim,
            commitment_cost=q.get("commitment_cost", 0.25),
            dead_code_threshold=q.get("dead_code_threshold", 1.0),
            decay=q.get("decay", 0.99),
            eps=q.get("eps", 1e-5),
            use_l2_norm=q.get("use_l2_norm", False),
        )

        # Projection layers between encoder/quantizer/decoder
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
            (logits [B, C, H, W], vq_result_dict)
        """
        z = self.encoder(x)                      # [B, num_latent, enc_dim]
        z = self.pre_quant(z)                    # [B, num_latent, token_dim]
        z_q, vq_result = self.quantizer(z)       # [B, num_latent, token_dim]
        z_q = self.post_quant(z_q)               # [B, num_latent, dec_dim]
        logits = self.decoder(z_q)               # [B, C, H, W]
        return logits, vq_result

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            indices: [B, num_latent] integer token ids
        """
        z = self.encoder(x)
        z = self.pre_quant(z)
        _, vq_result = self.quantizer(z)
        return vq_result["min_encoding_indices"]  # [B, num_latent]

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: [B, num_latent] integer token ids
        Returns:
            image: [B, C, H, W] reconstructed image, values in [0, 1]
        """
        z_q = self.quantizer.embedding(indices)  # [B, num_latent, token_dim]
        z_q = self.post_quant(z_q)               # [B, num_latent, dec_dim]
        logits = self.decoder(z_q)               # [B, C, H, W]
        return torch.sigmoid(logits)
