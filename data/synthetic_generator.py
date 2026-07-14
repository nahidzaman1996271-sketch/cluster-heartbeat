"""
Synthetic DCGM/Prometheus-style GPU telemetry generator.

Produces a realistic multi-GPU time series with:
  - normal idle/light/steady/bursty workload phases
  - a configurable fraction of GPUs that develop a "failure ramp"
    (rising temp, rising ECC/XID errors, degrading util) leading up
    to a simulated failure event, useful for training the failure /
    anomaly-detection models.

Used when real Prometheus/DCGM access or a public trace (Alibaba
Cluster Trace / Google Borg Trace) isn't available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import SETTINGS
from utils.logger import get_logger
from utils.schema import RAW_FEATURE_COLUMNS
from utils.helpers import set_global_seed, ensure_dir

log = get_logger("synthetic_generator")


def _gen_normal_series(n: int, base: float, noise: float, phase_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth-ish periodic + noisy series around `base`."""
    t = np.arange(n)
    seasonal = phase_scale * np.sin(2 * np.pi * t / max(1, n // 4))
    noise_term = rng.normal(0, noise, size=n)
    series = base + seasonal + noise_term
    return series


def _gen_failure_ramp(n: int, start_idx: int, rng: np.random.Generator):
    """Return multipliers (1.0 = normal) that ramp toward failure after start_idx."""
    ramp = np.ones(n)
    if start_idx < n:
        ramp_len = n - start_idx
        ramp[start_idx:] = 1.0 + np.linspace(0, 3.0, ramp_len) + rng.normal(0, 0.05, ramp_len)
    return ramp


def generate_gpu_trace(gpu_id: str, node_id: str, n_steps: int, interval_s: int,
                        will_fail: bool, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one GPU's full time series."""
    start_ts = pd.Timestamp("2026-01-01T00:00:00")
    timestamps = pd.date_range(start_ts, periods=n_steps, freq=f"{interval_s}s")

    gpu_util = np.clip(_gen_normal_series(n_steps, base=55, noise=12, phase_scale=25, rng=rng), 0, 100)
    mem_util = np.clip(gpu_util * 0.8 + rng.normal(0, 5, n_steps), 0, 100)
    cpu_util = np.clip(_gen_normal_series(n_steps, base=40, noise=10, phase_scale=15, rng=rng), 0, 100)
    ram_util = np.clip(cpu_util * 0.9 + rng.normal(0, 5, n_steps), 0, 100)

    gpu_temp = 45 + gpu_util * 0.35 + rng.normal(0, 2, n_steps)
    power_watts = 80 + gpu_util * 2.2 + rng.normal(0, 8, n_steps)

    ecc_errors = np.clip(rng.poisson(0.02, n_steps).astype(float), 0, None)
    xid_errors = np.clip(rng.poisson(0.01, n_steps).astype(float), 0, None)

    net_throughput = np.clip(_gen_normal_series(n_steps, base=300, noise=80, phase_scale=100, rng=rng), 0, None)
    disk_io = np.clip(_gen_normal_series(n_steps, base=50, noise=20, phase_scale=15, rng=rng), 0, None)

    job_runtime = np.clip(_gen_normal_series(n_steps, base=1800, noise=400, phase_scale=300, rng=rng), 0, None)
    queue_length = np.clip(rng.poisson(1.5, n_steps).astype(float), 0, None)
    active_processes = np.clip(rng.poisson(2.0, n_steps).astype(float) + 1, 1, None)

    if will_fail:
        start_idx = int(rng.integers(low=int(n_steps * 0.5), high=int(n_steps * 0.85)))
        ramp = _gen_failure_ramp(n_steps, start_idx, rng)
        gpu_temp = gpu_temp * np.clip(ramp, 1.0, 1.6)
        power_watts = power_watts * np.clip(ramp, 1.0, 1.3)
        ecc_errors = ecc_errors + np.clip((ramp - 1.0) * 20, 0, None)
        xid_errors = xid_errors + np.clip((ramp - 1.0) * 8, 0, None)
        gpu_util = np.clip(gpu_util * np.clip(2.0 - ramp, 0.2, 1.0), 0, 100)
        log.debug(f"GPU {gpu_id}: injected failure ramp starting at step {start_idx}")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "gpu_id": gpu_id,
        "node_id": node_id,
        "gpu_util": gpu_util,
        "mem_util": mem_util,
        "gpu_temp": gpu_temp,
        "power_watts": power_watts,
        "ecc_errors": ecc_errors,
        "xid_errors": xid_errors,
        "cpu_util": cpu_util,
        "ram_util": ram_util,
        "net_throughput_mbps": net_throughput,
        "disk_io_mbps": disk_io,
        "job_runtime_s": job_runtime,
        "queue_length": queue_length,
        "active_processes": active_processes,
    })
    return df[["timestamp", "gpu_id", "node_id"] + RAW_FEATURE_COLUMNS]


def generate_cluster_trace(num_gpus: int | None = None, num_days: int | None = None,
                            interval_s: int | None = None, failure_rate: float | None = None,
                            seed: int = 42) -> pd.DataFrame:
    """Generate a full synthetic cluster trace across many GPUs."""
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    cfg = SETTINGS["data"]["synthetic"]
    num_gpus = num_gpus or cfg["num_gpus"]
    num_days = num_days or cfg["num_days"]
    interval_s = interval_s or cfg["sample_interval_seconds"]
    failure_rate = failure_rate if failure_rate is not None else cfg["failure_rate"]

    n_steps = int((num_days * 24 * 3600) / interval_s)
    log.info(f"Generating synthetic trace: {num_gpus} GPUs x {n_steps} steps "
             f"({num_days}d @ {interval_s}s), failure_rate={failure_rate}")

    n_nodes = max(1, num_gpus // 8)
    frames = []
    for i in range(num_gpus):
        gpu_id = f"gpu-{i:03d}"
        node_id = f"node-{i % n_nodes:02d}"
        will_fail = rng.random() < failure_rate
        frames.append(generate_gpu_trace(gpu_id, node_id, n_steps, interval_s, will_fail, rng))

    trace = pd.concat(frames, ignore_index=True)
    log.info(f"Generated {len(trace)} rows across {num_gpus} GPUs")
    return trace


def generate_and_save(output_path: str | None = None, **kwargs) -> str:
    df = generate_cluster_trace(**kwargs)
    if output_path is None:
        out_dir = ensure_dir(SETTINGS["paths"]["raw_data_dir"])
        output_path = str(out_dir / "synthetic_trace.csv")
    df.to_csv(output_path, index=False)
    log.info(f"Saved synthetic trace to {output_path}")
    return output_path


if __name__ == "__main__":
    path = generate_and_save()
    print(f"Synthetic trace written to: {path}")
