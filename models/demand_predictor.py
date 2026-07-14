"""
LSTM-based resource demand forecaster: given the last `lookback` timesteps
of raw features, predicts gpu_util and mem_util for the next `horizon`
timesteps. Used for scheduling recommendations and cost optimization
(predicting when a GPU will be idle vs. busy).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config.settings import SETTINGS


class DemandPredictor(nn.Module):
    def __init__(self, num_features: int, horizon: int | None = None,
                 hidden_dim: int | None = None, num_layers: int | None = None,
                 num_targets: int = 2):
        super().__init__()
        cfg = SETTINGS["model"]["demand_predictor"]
        horizon = horizon or cfg["horizon"]
        hidden_dim = hidden_dim or cfg["hidden_dim"]
        num_layers = num_layers or cfg["num_layers"]

        self.horizon = horizon
        self.num_targets = num_targets

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, num_features)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        out = self.head(last_hidden)  # (batch, horizon * num_targets)
        return out.reshape(-1, self.horizon, self.num_targets)
