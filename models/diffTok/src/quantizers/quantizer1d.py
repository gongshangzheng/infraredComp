import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kmeans_pp import kmeans_plus_plus_init



class VectorQuantizer1d(nn.Module):
    """
    Improved 1D Vector Quantizer (EMA version)

    Input:
        z : [B, N, D]

    Output:
        z_q : [B, N, D]
    """

    def __init__(
        self,
        codebook_size: int,
        token_dim: int,
        commitment_cost: float = 0.25,
        dead_code_threshold: float = 1.0,
        decay: float = 0.99,
        eps: float = 1e-5,
        use_l2_norm: bool = False,
    ):
        super().__init__()

        self.codebook_size = codebook_size
        self.token_dim = token_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps
        self.use_l2_norm = use_l2_norm
        self.dead_code_threshold = dead_code_threshold

        # codebook
        self.embedding = nn.Embedding(codebook_size, token_dim)

        bound = 1 / math.sqrt(token_dim)
        self.embedding.weight.data.uniform_(-bound, bound)

        # EMA buffers
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, token_dim))

        # usage tracking
        self.register_buffer("embed_prob", torch.zeros(codebook_size))

    # ------------------------------------------------
    # optional kmeans init
    # ------------------------------------------------

    def compute_perplexity(self):
        """
        Compute perplexity of codebook usage using EMA-maintained probabilities.

        Higher perplexity indicates more uniform usage of all codebook entries.
        Perplexity ranges from 1 to self.codebook_size.

        Returns:
            perplexity: Scalar tensor representing perplexity value
        """
        # Use EMA-maintained embed_prob for stable perplexity computation
        probs = self.embed_prob / self.embed_prob.sum()
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())

        return perplexity

    def initialize_codebook(self, samples: torch.Tensor):

        with torch.no_grad():
            centroids = kmeans_plus_plus_init(
                samples=samples,
                n_clusters=self.codebook_size,
                device=self.embedding.weight.device,
            )

            self.embedding.weight.data.copy_(centroids)
            self.embed_avg.copy_(centroids)

    # ------------------------------------------------
    # forward
    # ------------------------------------------------

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, z):

        z = z.float()  # [B N D]
        z_for_st = z

        B, N, D = z.shape

        z_flat = z.reshape(-1, D)  # [BN D]

        # -----------------------------------------
        # cosine VQ (optional)
        # -----------------------------------------

        if self.use_l2_norm:
            z_flat = F.normalize(z_flat, dim=1)
            embedding = F.normalize(self.embedding.weight, dim=1)
            z_for_loss = F.normalize(z_for_st, dim=-1)
        else:
            embedding = self.embedding.weight
            z_for_loss = z_for_st

        # -----------------------------------------
        # compute distances
        # -----------------------------------------

        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            + embedding.pow(2).sum(1)
            - 2 * z_flat @ embedding.t()
        )  # [BN K]

        # -----------------------------------------
        # nearest code
        # -----------------------------------------

        indices = torch.argmin(dist, dim=1)  # [BN]

        z_q = embedding[indices]  # [BN D]

        # -----------------------------------------
        # reshape
        # -----------------------------------------

        z_q = z_q.view(B, N, D)
        indices = indices.view(B, N)

        # -----------------------------------------
        # losses
        # -----------------------------------------

        commitment_loss = self.commitment_cost * torch.mean(
            (z_q.detach() - z_for_loss) ** 2,
            dim=(1, 2),
        ).mean()

        codebook_loss = torch.mean(
            (z_q - z_for_loss.detach()) ** 2,
            dim=(1, 2),
        ).mean()

        vq_loss = commitment_loss + codebook_loss

        # -----------------------------------------
        # EMA update
        # -----------------------------------------

        if self.training:

            with torch.no_grad():

                cluster_size = torch.zeros(self.codebook_size, device=z.device)
                cluster_size.scatter_add_(0, indices.view(-1), torch.ones_like(indices.view(-1), dtype=z.dtype))

                embed_sum = torch.zeros(self.codebook_size, D, device=z.device)
                embed_sum.index_add_(0, indices.view(-1), z_flat)

                # -------------------------
                # EMA update
                # -------------------------

                self.cluster_size.mul_(self.decay).add_(
                    cluster_size, alpha=1 - self.decay
                )

                self.embed_avg.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )

                # -------------------------
                # normalize embeddings
                # -------------------------

                n = self.cluster_size.sum()

                cluster_size = (
                    (self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )

                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

                self.embedding.weight.data.copy_(embed_normalized)

                # -------------------------
                # dead code reset
                # -------------------------

                dead_mask = self.cluster_size < self.dead_code_threshold

                if dead_mask.any():

                    num_dead = dead_mask.sum()

                    rand_indices = torch.randint(
                        0,
                        z_flat.shape[0],
                        (num_dead,),
                        device=z_flat.device
                    )

                    new_vectors = z_flat[rand_indices]

                    self.embedding.weight.data[dead_mask] = new_vectors
                    self.embed_avg[dead_mask] = new_vectors
                    self.cluster_size[dead_mask] = self.dead_code_threshold
                # --------------------------------
                # USAGE STATS
                # --------------------------------
                counts = torch.bincount(
                    indices.view(-1),
                    minlength=self.codebook_size
                ).to(z.dtype)

                avg_probs = counts / counts.sum()

                self.embed_prob.mul_(self.decay).add_(
                    avg_probs, alpha=1 - self.decay
                )
        # -----------------------------------------
        # straight through
        # -----------------------------------------

        z_q = z_for_st + (z_q - z_for_st).detach()

        # -----------------------------------------
        # stats
        # -----------------------------------------

        num_indices = indices.numel()

        bits_per_index = math.log2(self.codebook_size)
        total_bits = num_indices * bits_per_index

        # Compute usage stats using EMA-maintained probabilities
        num_used = (self.embed_prob > 1e-6).sum()

        # Compute perplexity using EMA-maintained probabilities
        perplexity = self.compute_perplexity()

        result_dict = dict(
            quantizer_loss=vq_loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=indices,
            bits=total_bits,
            num_used_entries=num_used,
            codebook_usage_rate=num_used.float() / self.codebook_size,
            perplexity=perplexity,
        )

        return z_q, result_dict
