"""
PyTorch classifier that takes an embedding (from the autoencoder or PCA)
plus window summary stats and predicts a workload behavior class:
idle / light / steady / bursty.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config.settings import SETTINGS
from utils.schema import WORKLOAD_CLASSES


class WorkloadClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int | None = None, hidden_dim: int | None = None):
        super().__init__()
        cfg = SETTINGS["model"]["workload_classifier"]
        num_classes = num_classes or cfg["num_classes"]
        hidden_dim = hidden_dim or cfg["hidden_dim"]

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits; use CrossEntropyLoss / softmax externally

    @staticmethod
    def class_names():
        return WORKLOAD_CLASSES


def heuristic_labels_from_stats(mean_util: "np.ndarray") -> "np.ndarray":
    """
    Weak-supervision heuristic used to bootstrap training labels from
    gpu_util_mean when no ground-truth workload labels exist:
      util < 5%        -> idle
      5% <= util < 30%  -> light
      30% <= util < 70% -> steady
      util >= 70%       -> bursty
    """
    import numpy as np
    labels = np.zeros_like(mean_util, dtype=np.int64)
    labels[(mean_util >= 5) & (mean_util < 30)] = 1
    labels[(mean_util >= 30) & (mean_util < 70)] = 2
    labels[mean_util >= 70] = 3
    return labels
