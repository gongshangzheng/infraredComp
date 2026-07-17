import torch
def kmeans_plus_plus_init(
    samples: torch.Tensor,
    n_clusters: int,
    device: torch.device | None = None,
):
    """
    K-means++ initialization.

    Args:
        samples: [N, D] sample features
        n_clusters: number of clusters
        device: device for computation

    Returns:
        centroids: [K, D]
    """

    if device is None:
        device = samples.device

    samples = samples.to(device)

    N, D = samples.shape
    K = n_clusters

    centroids = torch.empty(K, D, device=device)

    # ------------------------------------------------
    # 1. choose first centroid randomly
    # ------------------------------------------------

    idx = torch.randint(0, N, (1,), device=device)
    centroids[0] = samples[idx]

    # ------------------------------------------------
    # 2. distances to nearest centroid
    # ------------------------------------------------

    dist = torch.sum((samples - centroids[0]) ** 2, dim=1)

    # ------------------------------------------------
    # 3. iterative selection
    # ------------------------------------------------

    for k in range(1, K):

        probs = dist / torch.sum(dist)

        next_idx = torch.multinomial(probs, 1)

        centroids[k] = samples[next_idx]

        # update nearest distance
        new_dist = torch.sum((samples - centroids[k]) ** 2, dim=1)

        dist = torch.minimum(dist, new_dist)

    return centroids
