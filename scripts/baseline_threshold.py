#!/usr/bin/env python
"""Per-feature threshold baseline — the Prometheus/DCGM-style detector this
paper positions itself against (Section II-A / Table III).

Calibrates a per-feature "normal" band (mean + k*std) from TRAIN windows
only, then scores each validation window by the fraction of features whose
window-max exceeds its own threshold. This is exactly the mental model of a
threshold alert per metric — it never looks at cross-feature correlation,
which is the gap FingerprintNet is designed to close (Section III-B).

Uses the *same* ingestion, windowing, and node-level train/val split (same
config, same seed) as scripts/train.py, and is scored with the exact same
anomaly_metrics / failure_lead_times functions as FingerprintNet, so the
numbers in reports/baseline_threshold_metrics.json are directly comparable
to checkpoints/best/metrics.json.

Usage::

    python scripts/baseline_threshold.py --config configs/default.yaml
    python scripts/baseline_threshold.py --k 2.5   # sweep the calibration knob
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_heartbeat.config import load_config, load_feature_registry
from cluster_heartbeat.data.ingestion import ingest
from cluster_heartbeat.data.preprocessing import clean_all
from cluster_heartbeat.data.windows import build_windows, concat_batches
from cluster_heartbeat.training.metrics import anomaly_metrics, failure_lead_times
from cluster_heartbeat.utils.helpers import set_seed, write_json


def make_splits(cfg, result, feature_names, classes):
    """Mirror Trainer._make_splits exactly (same seed -> same nodes)."""
    rng = np.random.default_rng(cfg.seed)
    nodes = sorted(result.frames)
    rng.shuffle(nodes)
    n_val = max(1, int(round(cfg.train.val_split * len(nodes))))
    val_nodes, train_nodes = set(nodes[:n_val]), set(nodes[n_val:])

    failing = set(result.failures)
    if failing and not (val_nodes & failing):
        swap_in = sorted(failing & train_nodes)[:1]
        if swap_in:
            swap_out = sorted(val_nodes - failing)[:1]
            train_nodes -= set(swap_in)
            val_nodes |= set(swap_in)
            if swap_out:
                val_nodes -= set(swap_out)
                train_nodes |= set(swap_out)

    def build(node_ids):
        batches = []
        for node in sorted(node_ids):
            failure = result.failures.get(node)
            batches.append(build_windows(
                node_id=node, df=result.frames[node], feature_names=feature_names,
                classes=classes, size=cfg.window.size, stride=cfg.window.stride,
                interval_seconds=cfg.data.synthetic.interval_seconds,
                ttf_horizon_hours=cfg.window.ttf_horizon_hours,
                failure_step=failure["failure_step"] if failure else None,
            ))
        return concat_batches(batches)

    return build(train_nodes), build(val_nodes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features", default="configs/features.yaml")
    ap.add_argument("--k", type=float, default=3.0,
                     help="per-feature threshold = mean + k*std, calibrated on train")
    ap.add_argument("--out", default="reports/baseline_threshold_metrics.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    specs, classes = load_feature_registry(args.features)
    feature_names = [f.name for f in specs]
    set_seed(cfg.seed)

    result = ingest(cfg.data.source, cfg, specs)
    result.frames = clean_all(result.frames, specs, cfg.data.synthetic.interval_seconds)

    train_batch, val_batch = make_splits(cfg, result, feature_names, classes)

    # --- calibrate one threshold per physical feature, from train only -----
    mean = train_batch.X.mean(axis=(0, 2))
    std = np.maximum(train_batch.X.std(axis=(0, 2)), 1e-6)
    thresholds = mean + args.k * std

    def score(batch):
        window_max = batch.X.max(axis=2)                      # (N, F)
        return (window_max > thresholds[None, :]).mean(axis=1).astype(np.float32)

    scores = score(val_batch)
    anomaly = anomaly_metrics(scores, val_batch.anomaly_gt)

    end_steps = np.array([m["end_step"] for m in val_batch.meta])
    node_ids = np.array([m["node_id"] for m in val_batch.meta])
    alert_thr = anomaly.get("best", {}).get("threshold", 0.5)
    lead_times = {}
    for node, failure in result.failures.items():
        mask = node_ids == node
        if mask.sum() == 0:
            continue
        lt = failure_lead_times(
            scores[mask], end_steps[mask], failure["failure_step"],
            alert_thr, cfg.data.synthetic.interval_seconds,
        )
        if lt is not None:
            lead_times[node] = lt

    metrics = {
        "backend": "threshold_baseline",
        "k": args.k,
        "per_feature_threshold": dict(zip(feature_names, thresholds.tolist())),
        "n_val_windows": int(len(val_batch)),
        "n_val_nodes": int(len(set(node_ids))),
        "anomaly": anomaly,
        "predictive_lead_time_hours": {
            "mean": float(np.mean(list(lead_times.values()))) if lead_times else None,
            "per_node": lead_times,
        },
    }

    print(json.dumps(metrics, indent=2, default=str))
    out_path = write_json(metrics, Path(args.out))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()