"""
Trains the WorkloadClassifier on top of frozen PCA embeddings + window
summary stats, using heuristic labels derived from mean GPU utilization
(idle/light/steady/bursty). Swap in real labels if available.
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
from features.windowing import make_windows, window_summary_stats
from features.normalization import WindowScaler
from models.pca_embedding import PCAEmbedder
from models.workload_classifier import WorkloadClassifier, heuristic_labels_from_stats

log = get_logger("train_classifier")


def train(epochs: int | None = None, batch_size: int | None = None, lr: float | None = None,
          val_split: float = 0.15) -> str:
    set_global_seed(42)
    cfg = SETTINGS["model"]["workload_classifier"]
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]
    lr = lr or cfg["lr"]

    device = get_device()
    log.info(f"Training workload classifier on device={device} for {epochs} epochs")

    clean_path = Path(SETTINGS["paths"]["processed_data_dir"]) / "clean_trace.csv"
    df = load_trace(clean_path)
    windows, _, _ = make_windows(df)

    scaler = WindowScaler.load()
    scaled = scaler.transform(windows)

    embedder = PCAEmbedder.load()
    z = embedder.transform(scaled)  # (N, latent_dim)

    stats = window_summary_stats(windows)
    mean_util = stats["gpu_util_mean"].to_numpy()
    labels = heuristic_labels_from_stats(mean_util)

    features = np.concatenate([z, stats.to_numpy()], axis=1).astype(np.float32)

    n = len(features)
    idx = np.random.permutation(n)
    n_val = int(n * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train = torch.tensor(features[train_idx])
    y_train = torch.tensor(labels[train_idx], dtype=torch.long)
    X_val = torch.tensor(features[val_idx])
    y_val = torch.tensor(labels[val_idx], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = WorkloadClassifier(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt_path = Path(SETTINGS["paths"]["checkpoint_dir"]) / "workload_classifier.pt"
    ensure_dir(ckpt_path.parent)
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / max(total, 1)
        log.info(f"Epoch {epoch}/{epochs} - val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": features.shape[1],
                "val_acc": acc,
            }, ckpt_path)

    log.info(f"Training complete. Best val_acc={best_acc:.4f}. Checkpoint: {ckpt_path}")
    return str(ckpt_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train(epochs=args.epochs)
