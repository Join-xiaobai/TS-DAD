import os
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pygod.utils import load_data
from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from torch_geometric.datasets import Planetoid

def load_benchmark_dataset(name, root_dir="./data"):
    """
    Unified benchmark dataset loader for all 8 datasets evaluated in TS-DAD.
    
    Supports:
    - Organic Real-World Graphs: 'books', 'disney', 'enron', 'weibo', 'reddit'
    - Synthetic Benchmark Graphs: 'cora', 'citeseer', 'pubmed' (10% standard perturbation)
    """
    dname = name.lower()
    os.makedirs(root_dir, exist_ok=True)
    
    if dname in ["books", "disney", "enron"]:
        data = load_data(dname)
        data.x = torch.tensor(MinMaxScaler().fit_transform(data.x.numpy()), dtype=torch.float)
        y_true = (data.y > 0).long().numpy()
        data.y = torch.tensor(y_true, dtype=torch.long)
        return data, y_true, "organic"
        
    elif dname in ["weibo", "reddit"]:
        data = load_data(dname)
        y_true = (data.y > 0).long().numpy()
        data.y = torch.tensor(y_true, dtype=torch.long)
        return data, y_true, "organic"
        
    elif dname in ["cora", "citeseer", "pubmed"]:
        p_name = dname.capitalize() if dname != "citeseer" else "CiteSeer"
        raw_data = Planetoid(root=os.path.join(root_dir, "Planetoid"), name=p_name)[0]
        cur_data = raw_data.clone().cpu()
        num_c = int(raw_data.num_nodes * 0.05)
        data_c, y_c = gen_contextual_outlier(cur_data, n=num_c, k=50)
        num_cliques = max(1, num_c // 15)
        data_s, y_s = gen_structural_outlier(data_c, m=num_cliques, n=15)
        y_true = (y_c | y_s).long().numpy()
        data_s.y = torch.tensor(y_true, dtype=torch.long)
        return data_s, y_true, "synthetic"
        
    else:
        raise ValueError(f"Unknown dataset '{name}'. Available: books, disney, enron, weibo, reddit, cora, citeseer, pubmed")
