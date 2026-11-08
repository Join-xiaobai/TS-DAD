import torch
import numpy as np
from sklearn.cluster import KMeans

class NormalPrototypeAnchor:
    """
    Self-Refined Normal Prototype Anchoring Module.
    
    Distills K manifold anchor centers from high-confidence inliers, providing
    adaptive global reference across varying node degree distributions.
    """
    def __init__(self, num_prototypes=5, inlier_ratio=0.9):
        self.num_prototypes = num_prototypes
        self.inlier_ratio = inlier_ratio
        self.prototypes = None

    def fit(self, z_attr, raw_scores):
        if isinstance(z_attr, torch.Tensor):
            z_attr_np = z_attr.detach().cpu().numpy()
        else:
            z_attr_np = z_attr

        threshold = np.percentile(raw_scores, self.inlier_ratio * 100)
        inlier_mask = raw_scores <= threshold
        inlier_reps = z_attr_np[inlier_mask]

        num_clusters = min(self.num_prototypes, len(inlier_reps))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(inlier_reps)
        self.prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=z_attr.device if isinstance(z_attr, torch.Tensor) else "cpu")

    def compute_global_deviation(self, z_attr):
        if self.prototypes is None:
            return torch.zeros(z_attr.size(0), device=z_attr.device)
        # Minimum squared distance to any prototype
        diff = z_attr.unsqueeze(1) - self.prototypes.unsqueeze(0)
        dists = torch.sum(diff ** 2, dim=-1)
        min_dists, _ = torch.min(dists, dim=1)
        return min_dists
