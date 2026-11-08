import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import numpy as np
from tsdad.dataset import load_benchmark_dataset
from tsdad.detector import TSDAD

def main():
    parser = argparse.ArgumentParser(description="Run TS-DAD on a single graph benchmark.")
    parser.add_argument("--dataset", type=str, default="books", help="Dataset name: books, disney, enron, weibo, reddit, cora, citeseer, pubmed")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.001, help="EODR weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset: {args.dataset}...")
    data, y_true, dtype = load_benchmark_dataset(args.dataset)
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}, Dim: {data.num_node_features}, Anomalies: {y_true.sum()} ({y_true.sum()/data.num_nodes*100:.2f}%)")

    gamma = 0.05 if args.dataset in ["weibo", "reddit"] else args.gamma
    beta = 0.6 if args.dataset == "weibo" else (0.3 if args.dataset == "reddit" else 0.5)

    detector = TSDAD(epochs=args.epochs, lr=args.lr, gamma=gamma, beta=beta)
    print("Training TS-DAD...")
    detector.fit(data)

    auc, ap = detector.evaluate(y_true)
    print(f"\n================ Result ({args.dataset.upper()}) ================")
    print(f"ROC-AUC : {auc * 100:.2f}%")
    print(f"AUPRC   : {ap * 100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    main()
