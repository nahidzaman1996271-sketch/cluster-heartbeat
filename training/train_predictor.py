"""
Trains the LSTM DemandPredictor: given `lookback` timesteps of the 13 raw
features, predict [gpu_util, mem_util] for the next `horizon` timesteps.
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
from features.normalization import WindowScaler
from models.demand_predictor import DemandPredictor
from utils.schema import RAW_FEATURE_COLUMNS

log = get_logger("train_predictor")

TARGET_IDX = [RAW_FEATURE_COLUMNS.index("gpu_util"), RAW_FEATURE_COLUMNS.index("mem_util")]


def make_supervised_sequences(df, lookback: int, horizon: int):
    """Build (X, y) sequence pairs per-GPU: X=lookback raw steps, y=next horizon [util,mem]."""
    X_list, y_list = [], []
    for gpu_id, group in df.groupby("gpu_id", sort=False):
        values = group.sort_values("timestamp")[RAW_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        n = len(values)
        for start in range(0, n - lookback - horizon + 1, max(1, lookback // 2)):
            X_list.append(values[start:start + lookback])
            y_list.append(values[start + lookback:start + lookback + horizon][:, TARGET_IDX])
    return np.stack(X_list), np.stack(y_list)


def train(epochs: int | None = None, batch_size: int | None = None, lr: float | None = None,
          val_split: float = 0.15) -> str:
    set_global_seed(42)
    cfg = SETTINGS["model"]["demand_predictor"]
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]
    lr = lr or cfg["lr"]
    lookback, horizon = cfg["lookback"], cfg["horizon"]

    device = get_device()
    log.info(f"Training demand predictor on device={device} for {epochs} epochs "
             f"(lookback={lookback}, horizon={horizon})")

    clean_path = Path(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    df = load_trace(clean_path)

    X, y = make_supervised_sequences(df, lookback, horizon)
    log.info(f"Built {len(X)} supervised sequences")

    scaler = WindowScaler.load()
    X_scaled = scaler.transform(X)

    n = len(X_scaled)
    idx = np.random.permutation(n)
    n_val = int(n * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled[train_idx]), torch.tensor(y[train_idx])),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled[val_idx]), torch.tensor(y[val_idx])),
        batch_size=batch_size, shuffle=False)

    model = DemandPredictor(num_features=X.shape[2], horizon=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    ckpt_path = Path(SETTINGS["paths"]["checkpoint_dir"]) / "demand_predictor.pt"
    ensure_dir(ckpt_path.parent)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        log.info(f"Epoch {epoch}/{epochs} - val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "num_features": X.shape[2],
                "horizon": horizon,
                "lookback": lookback,
                "val_loss": val_loss,
            }, ckpt_path)

    log.info(f"Training complete. Best val_loss={best_val:.5f}. Checkpoint: {ckpt_path}")
    return str(ckpt_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs)
