"""
PyTorch autoencoder that learns a compressed "workload fingerprint" (latent
embedding) from a flattened sliding window of the 13 raw features.

Input:  (batch, window_size, num_features) -> flattened to (batch, window_size*num_features)
Output: reconstruction of the same shape + the latent embedding.

The reconstruction error doubles as the primary anomaly-detection signal
(see models/anomaly_detector.py).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from config.settings import SETTINGS


class GPUAutoencoder(nn.Module):
    def __init__(self, window_size: int, num_features: int,
                 hidden_dims: List[int] | None = None, latent_dim: int | None = None):
        super().__init__()
        cfg = SETTINGS["model"]["autoencoder"]
        hidden_dims = hidden_dims or cfg["hidden_dims"]
        latent_dim = latent_dim or cfg["latent_dim"]

        self.window_size = window_size
        self.num_features = num_features
        input_dim = window_size * num_features

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.BatchNorm1d(h)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror of encoder)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.BatchNorm1d(h)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, num_features)
        flat = x.reshape(x.size(0), -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder(z)
        return flat.reshape(-1, self.window_size, self.num_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error, shape (batch,)."""
        recon, _ = self.forward(x)
        return torch.mean((recon - x) ** 2, dim=(1, 2))
