import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from tsdad.dataset import load_benchmark_dataset
from tsdad.detector import TSDAD

def main():
    datasets = ["books", "disney", "enron", "weibo", "reddit", "cora", "citeseer", "pubmed"]
    seeds = [42, 43, 44]

    print("================================================================")
    print("   TS-DAD Official Benchmark Reproduction (8 Datasets x 3 Seeds)")
    print("================================================================")

    for dname in datasets:
        auc_list, ap_list = [], []
        data, y_true, dtype = load_benchmark_dataset(dname)
        gamma = 0.05 if dname in ["weibo", "reddit"] else 0.001
        beta = 0.6 if dname == "weibo" else (0.3 if dname == "reddit" else 0.5)

        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            det = TSDAD(epochs=100, lr=0.005, gamma=gamma, beta=beta)
            det.fit(data)
            auc, ap = det.evaluate(y_true)
            auc_list.append(auc)
            ap_list.append(ap)

        mean_auc, std_auc = np.mean(auc_list) * 100, np.std(auc_list) * 100
        mean_ap, std_ap = np.mean(ap_list) * 100, np.std(ap_list) * 100
        print(f"[{dname.upper():8s}] ROC-AUC: {mean_auc:.2f} ± {std_auc:.2f}% | AUPRC: {mean_ap:.2f} ± {std_ap:.2f}%")

if __name__ == "__main__":
    main()
