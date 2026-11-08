import torch
import torch.nn.functional as F

def compute_eodr_loss(z_attr, z_topo):
    """
    Empirical Orthogonal Decoupling Regularization (EODR).
    
    Eliminates linear correlation redundancy between attribute and topological spaces:
    1. Explicit column-centering and standard deviation scaling.
    2. Minimizes the squared Frobenius norm of the cross-covariance matrix.
    """
    num_samples = z_attr.size(0)
    z_a_norm = (z_attr - z_attr.mean(dim=0, keepdim=True)) / (z_attr.std(dim=0, keepdim=True) + 1e-8)
    z_t_norm = (z_topo - z_topo.mean(dim=0, keepdim=True)) / (z_topo.std(dim=0, keepdim=True) + 1e-8)
    cov = torch.mm(z_a_norm.t(), z_t_norm) / num_samples
    return torch.norm(cov, p='fro') ** 2

def reconstruction_loss(x, x_rec):
    """Euclidean reconstruction residual per node."""
    return torch.mean(torch.sqrt(torch.sum((x - x_rec).pow(2), dim=1)))
