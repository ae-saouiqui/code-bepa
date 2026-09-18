
import torch


class RankMEMetric:

    def __call__(self,embeddings):
    # Compute SVD to get singular values
    # We only need singular values, not U and V matrices
        try:
            _, S, _ = torch.svd(embeddings)
        except RuntimeError:
            # Fallback if SVD fails (e.g., for very large matrices)
            # Use a more stable but slower method
            S = torch.linalg.svdvals(embeddings)
        # Normalize singular values to get a probability distribution
        p = (S / S.sum()) + 1e-10  # Add small epsilon for numerical stability
        # Compute Shannon entropy of the singular value distribution
        entropy = -(p * torch.log(p)).sum()
        # Effective rank is exp(entropy)
        # This gives a measure between 1 (completely collapsed) and min(N, d) (full rank)
        return torch.exp(entropy).item()