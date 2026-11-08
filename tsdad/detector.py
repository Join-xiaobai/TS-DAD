import torch
import numpy as np
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from .models import FeatureDecoupledGCN
from .loss import compute_eodr_loss, reconstruction_loss
from .prototype import NormalPrototypeAnchor

class TSDAD:
    """
    TS-DAD Anomaly Detector Class.
    
    Parameters:
    -----------
    hid_dim : int, hidden representation dimension (default: 64)
    embed_dim : int, embedding space dimension (default: 32)
    lr : float, learning rate for Adam optimizer (default: 0.005)
    epochs : int, number of training epochs (default: 100)
    alpha : float, weight for attribute loss (default: 0.5)
    gamma : float, weight for EODR orthogonal regularization (default: 0.001)
    beta : float, score balance parameter (default: 0.5)
    device : str, device identifier ('cuda' or 'cpu')
    """
    def __init__(self, hid_dim=64, embed_dim=32, lr=0.005, epochs=100, 
                 alpha=0.5, gamma=0.001, beta=0.5, device=None):
        self.hid_dim = hid_dim
        self.embed_dim = embed_dim
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.decision_scores_ = None

    def fit(self, data):
        num_nodes = data.num_nodes
        in_dim = data.num_node_features
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        deg = torch.bincount(data.edge_index[0], minlength=num_nodes).numpy()

        self.model = FeatureDecoupledGCN(in_dim, hid_dim=self.hid_dim, embed_dim=self.embed_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        is_dense = (num_nodes <= 3000)
        if is_dense:
            dense_adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0].to(self.device)

        self.model.train()
        for ep in range(self.epochs):
            optimizer.zero_grad()
            z_a, z_t, x_rec, adj_rec = self.model(x, edge_index)

            l_attr = reconstruction_loss(x, x_rec)

            if is_dense:
                l_struct = torch.mean(torch.sqrt(torch.sum((dense_adj - adj_rec).pow(2), dim=1)))
            else:
                pos_u, pos_v = edge_index[:, :min(50000, edge_index.size(1))]
                pos_scores = (z_t[pos_u] * z_t[pos_v]).sum(dim=1)
                neg_v = torch.randint(0, num_nodes, (pos_u.size(0),), device=self.device)
                neg_scores = (z_t[pos_u] * z_t[neg_v]).sum(dim=1)
                l_struct = F.mse_loss(pos_scores, torch.ones_like(pos_scores)) + F.mse_loss(neg_scores, torch.zeros_like(neg_scores))

            l_ortho = compute_eodr_loss(z_a, z_t)
            total_loss = self.alpha * l_attr + (1.0 - self.alpha) * l_struct + self.gamma * l_ortho
            total_loss.backward()
            optimizer.step()

        # Decision scoring
        self.model.eval()
        with torch.no_grad():
            z_a, z_t, x_rec, adj_rec = self.model(x, edge_index)
            e_a = torch.sqrt(torch.sum((x - x_rec).pow(2), dim=1)).cpu().numpy()
            pred_deg = torch.norm(z_t, dim=1).cpu().numpy()
            true_deg_norm = deg / (deg.max() + 1e-8)

            if is_dense:
                e_s = torch.sqrt(torch.sum((dense_adj - adj_rec).pow(2), dim=1)).cpu().numpy()
            else:
                e_s = np.abs(pred_deg - true_deg_norm)

            e_a_n = (e_a - e_a.min()) / (e_a.max() - e_a.min() + 1e-8)
            e_s_n = (e_s - e_s.min()) / (e_s.max() - e_s.min() + 1e-8)
            self.decision_scores_ = self.beta * e_a_n + (1.0 - self.beta) * e_s_n

        return self

    def decision_function(self, data=None):
        return self.decision_scores_

    def evaluate(self, y_true):
        auc = roc_auc_score(y_true, self.decision_scores_)
        ap = average_precision_score(y_true, self.decision_scores_)
        return auc, ap
