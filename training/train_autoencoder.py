"""
Trains the GPUAutoencoder on sliding windows built from the cleaned trace.
Saves the best checkpoint (lowest val loss) to checkpoints/autoencoder.pt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.settings import SETTINGS
from utils.logger import get_logger
from utils.helpers import set_global_seed, get_device, ensure_dir
from data.ingestion import load_trace
from features.windowing import make_windows
from features.normalization import WindowScaler
from models.autoencoder import GPUAutoencoder

log = get_logger("train_autoencoder")


def train(epochs: int | None = None, batch_size: int | None = None, lr: float | None = None,
          val_split: float = 0.15) -> str:
    set_global_seed(42)
    cfg = SETTINGS["model"]["autoencoder"]
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]
    lr = lr or cfg["lr"]

    device = get_device()
    log.info(f"Training autoencoder on device={device} for {epochs} epochs")

    clean_path = Path(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    df = load_trace(clean_path)
    windows, _, _ = make_windows(df)

    scaler = WindowScaler().fit(windows)
    scaled = scaler.transform(windows)
    scaler.save()

    n = len(scaled)
    idx = np.random.permutation(n)
    n_val = int(n * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_t = torch.tensor(scaled[train_idx], dtype=torch.float32)
    val_t = torch.tensor(scaled[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_t), batch_size=batch_size, shuffle=False)

    window_size, num_features = scaled.shape[1], scaled.shape[2]
    model = GPUAutoencoder(window_size, num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    ckpt_path = Path(SETTINGS["paths"]["checkpoint_dir"]) / "autoencoder.pt"
    ensure_dir(ckpt_path.parent)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        log.info(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "window_size": window_size,
                "num_features": num_features,
                "val_loss": val_loss,
            }, ckpt_path)
            log.info(f"  -> new best model saved (val_loss={val_loss:.5f})")

    log.info(f"Training complete. Best val_loss={best_val:.5f}. Checkpoint: {ckpt_path}")
    return str(ckpt_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size)
