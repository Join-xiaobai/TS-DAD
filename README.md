# TS-DAD: Attribute-Preserving Topology-Decoupled Learning for Attributed Graph Anomaly Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **TS-DAD** (*Attribute-Preserving Topology-Decoupled Learning for Real-World Attributed Graph Anomaly Detection*).

---

## 🌟 Method Overview

Graph Neural Network (GNN) autoencoders experience catastrophic degradation on real-world attributed anomaly detection benchmarks due to **Feature Dilution**: recursive neighborhood aggregation acts as a low-pass filter, averaging rare outlier features into normative backgrounds.

**TS-DAD** resolves this bottleneck through:
1. **Attribute-Preserving Stream**: Completely isolates attribute reconstruction from graph convolutions via an unpooled multi-layer perceptron pathway.
2. **Dedicated Topological Stream**: Explicitly models connectivity anomalies and structural deviations via a dedicated GCN pathway.
3. **Empirical Orthogonal Decoupling Regularization (EODR)**: Enforces statistical independence and zero cross-covariance between attribute and topological latent spaces.
4. **Self-Refined Normal Prototype Anchoring**: Distills global manifold centroids from high-confidence inliers to adaptively calibrate localized residuals across varying graph density regimes.

---

## 📊 Benchmark Results

Evaluated across **8 benchmark datasets** against 16 state-of-the-art baselines under dual evaluation metrics:

| Dataset | Type | Nodes | Edges | Anomalies | ROC-AUC (%) | AUPRC (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Amazon Books** | Organic | 1,418 | 3,695 | 28 (1.97%) | **88.29 ± 0.61** | **31.32 ± 0.84** |
| **Amazon Disney** | Organic | 124 | 333 | 6 (4.84%) | **91.85 ± 0.52** | **44.15 ± 0.76** |
| **Amazon Enron** | Organic | 13,533 | 176,442 | 5 (0.04%) | **79.52 ± 0.85** | **18.74 ± 0.92** |
| **Weibo** | Organic | 8,405 | 407,963 | 868 (10.3%) | **99.64 ± 0.08** | **96.85 ± 0.12** |
| **Reddit** | Organic | 10,984 | 168,016 | 366 (3.33%) | **71.30 ± 0.42** | **12.48 ± 0.35** |
| **Cora** | Synthetic | 2,708 | 5,429 | 150 (5.54%) | **92.14 ± 0.38** | **68.25 ± 0.55** |
| **CiteSeer** | Synthetic | 3,327 | 4,732 | 150 (4.51%) | **91.45 ± 0.45** | **65.80 ± 0.62** |
| **PubMed** | Synthetic | 19,717 | 44,338 | 600 (3.04%) | **95.62 ± 0.25** | **78.40 ± 0.48** |

*Efficiency Highlight*: Delivers up to **353× empirical training speedup** over message-passing graph autoencoders.

---

## 🚀 Installation

```bash
git clone https://github.com/Join-xiaobai/TS-DAD.git
cd TS-DAD
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ Quickstart

Run TS-DAD on any dataset:
```bash
# Organic benchmark
python scripts/run_single.py --dataset books

# Synthetic benchmark
python scripts/run_single.py --dataset cora
```

Reproduce full benchmarks across all 8 datasets:
```bash
python scripts/run_benchmark.py
```

---

## 📑 Citation

```bibtex
@article{tsdad,
  title={Attribute-Preserving Topology-Decoupled Learning for Real-World Attributed Graph Anomaly Detection},
  author={Anonymous},
  journal={Preprint}
}
```

---

## 📄 License
This project is released under the [MIT License](LICENSE).
