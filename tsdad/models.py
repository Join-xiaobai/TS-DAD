import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FeatureDecoupledGCN(nn.Module):
    """
    Core TS-DAD Network Architecture.
    
    1. Attribute Stream: Multi-layer perceptron (MLP) without graph convolutions,
       strictly preserving high-dimensional outlier feature margins without neighborhood dilution.
    2. Topological Structure Stream: Graph Convolutional Network (GCN) that explicitly
       models relational adjacency and network density patterns.
    3. Latent Outputs:
       - z_attr: attribute latent representation
       - z_topo: topological latent representation
       - x_rec: reconstructed node attributes
       - adj_rec: reconstructed inner-product adjacency
    """
    def __init__(self, in_dim, hid_dim=64, embed_dim=32, dropout=0.1):
        super(FeatureDecoupledGCN, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.embed_dim = embed_dim
        self.dropout = dropout

        # 1. Attribute-preserving Stream (Convolution-free MLP)
        self.attr_enc1 = nn.Linear(in_dim, hid_dim)
        self.attr_enc2 = nn.Linear(hid_dim, embed_dim)
        self.attr_dec1 = nn.Linear(embed_dim, hid_dim)
        self.attr_dec2 = nn.Linear(hid_dim, in_dim)

        # 2. Structural Topology Stream (GCN)
        self.topo_enc1 = GCNConv(in_dim, hid_dim)
        self.topo_enc2 = GCNConv(hid_dim, embed_dim)

    def forward(self, x, edge_index):
        # Attribute Stream
        h_a = F.relu(self.attr_enc1(x))
        h_a = F.dropout(h_a, p=self.dropout, training=self.training)
        z_attr = self.attr_enc2(h_a)

        h_a_dec = F.relu(self.attr_dec1(z_attr))
        h_a_dec = F.dropout(h_a_dec, p=self.dropout, training=self.training)
        x_rec = self.attr_dec2(h_a_dec)

        # Topological Stream
        h_t = F.relu(self.topo_enc1(x, edge_index))
        h_t = F.dropout(h_t, p=self.dropout, training=self.training)
        z_topo = self.topo_enc2(h_t, edge_index)

        adj_rec = torch.mm(z_topo, z_topo.t())

        return z_attr, z_topo, x_rec, adj_rec
