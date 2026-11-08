"""
TS-DAD: Attribute-Preserving Topology-Decoupled Learning for Attributed Graph Anomaly Detection
"""

from .models import FeatureDecoupledGCN
from .loss import compute_eodr_loss, reconstruction_loss
from .prototype import NormalPrototypeAnchor
from .detector import TSDAD
from .dataset import load_benchmark_dataset

__version__ = "1.0.0"
__all__ = [
    "TSDAD",
    "FeatureDecoupledGCN",
    "compute_eodr_loss",
    "reconstruction_loss",
    "NormalPrototypeAnchor",
    "load_benchmark_dataset",
]
