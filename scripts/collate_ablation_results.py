#!/usr/bin/env python
"""Collect metrics.json from the multi-task model + 4 single-task ablations
+ the threshold baseline into one table for the paper (Section IV-D).

Run after:
    python scripts/train.py --config configs/default.yaml                      # multi-task (already have this)
    python scripts/train.py --config configs/ablation/reconstruction_only.yaml
    python scripts/train.py --config configs/ablation/classification_only.yaml
    python scripts/train.py --config configs/ablation/demand_only.yaml
    python scripts/train.py --config configs/ablation/ttf_only.yaml
    python scripts/baseline_threshold.py --config configs/default.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "FingerprintNet (multi-task)": ROOT / "checkpoints/best/metrics.json",
    "Single-task: reconstruction only": ROOT / "checkpoints/ablation_reconstruction_only/best/metrics.json",
    "Single-task: classification only": ROOT / "checkpoints/ablation_classification_only/best/metrics.json",
    "Single-task: demand only": ROOT / "checkpoints/ablation_demand_only/best/metrics.json",
    "Single-task: TTF only": ROOT / "checkpoints/ablation_ttf_only/best/metrics.json",
}
BASELINE = ROOT / "reports/baseline_threshold_metrics.json"


def get(d, *path, default="n/a"):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def main() -> None:
    rows = []
    for label, path in RUNS.items():
        if not path.exists():
            rows.append((label, "not run yet", "", "", ""))
            continue
        m = json.loads(path.read_text())
        rows.append((
            label,
            f"{get(m, 'anomaly', 'auroc'):.3f}" if isinstance(get(m, 'anomaly', 'auroc'), float) else "n/a",
            f"{get(m, 'anomaly', 'best', 'f1'):.3f}" if isinstance(get(m, 'anomaly', 'best', 'f1'), float) else "n/a",
            f"{get(m, 'classification', 'accuracy'):.3f}" if isinstance(get(m, 'classification', 'accuracy'), float) else "n/a",
            f"{get(m, 'ttf_hours', 'mae'):.2f}" if isinstance(get(m, 'ttf_hours', 'mae'), float) else "n/a",
        ))

    print(f"{'Model':<38} {'AUROC':>8} {'F1':>8} {'Cls Acc':>8} {'TTF MAE(h)':>11}")
    print("-" * 75)
    for r in rows:
        print(f"{r[0]:<38} {r[1]:>8} {r[2]:>8} {r[3]:>8} {r[4]:>11}")

    if BASELINE.exists():
        b = json.loads(BASELINE.read_text())
        auroc = get(b, "anomaly", "auroc")
        f1 = get(b, "anomaly", "best", "f1")
        print("-" * 75)
        print(f"{'Threshold baseline (Prometheus-style)':<38} "
              f"{auroc:>8.3f} {f1:>8.3f} {'n/a':>8} {'n/a':>11}")
    else:
        print("\n(threshold baseline not run yet — see scripts/baseline_threshold.py)")


if __name__ == "__main__":
    main()
