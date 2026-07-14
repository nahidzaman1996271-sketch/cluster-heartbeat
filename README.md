# Cluster Heartbeat

AI-powered GPU cluster monitoring system. Ingests Prometheus/DCGM-style
telemetry, builds a unified per-GPU **workload fingerprint** (embedding +
anomaly score), and powers three services from that single fingerprint:

1. **Predictive Failure Detection** — health score, failure risk score, estimated time to failure
2. **Smart GPU Scheduling** — ranks GPUs for new job placement, flags GPUs to drain
3. **GPU Cost Optimization** — detects idle GPUs, estimates savings

Outputs dashboard-ready JSON for Grafana (JSON API datasource) or a custom React dashboard.

## Project layout

```
config/        YAML config + loader (config.yaml, settings.py)
utils/         logger, shared schema (13 feature columns), helpers
data/          synthetic generator, Prometheus/file ingestion, cleaning
features/      sliding-window generation, per-window stats, normalization
models/        PyTorch autoencoder, PCA embedder, IsolationForest anomaly
               detector, workload classifier, LSTM demand predictor
training/      train_autoencoder.py, train_classifier.py, train_predictor.py
inference/     end-to-end pipeline + SHAP/formula-based explainability
services/      health_score.py, scheduler.py, cost_optimizer.py
dashboards/    JSON payload builder for Grafana/React
api/           FastAPI app (main.py) + pydantic schemas
tests/         pytest smoke tests covering the full pipeline
checkpoints/   saved scaler / embedder / anomaly detector / model weights
```

## Setup

```bash
pip install -r requirements.txt
```

## Quickstart

```bash
# 1. Generate synthetic DCGM/Prometheus-style data (or point data/ingestion.py at real data)
python -m data.synthetic_generator

# 2. Clean it
python -m data.preprocessing

# 3. Fit the feature scaler + PCA embedder + anomaly detector
#    (fast, non-deep-learning path - good default)
python -c "
from pathlib import Path
from config.settings import SETTINGS
from data.ingestion import load_trace
from features.windowing import make_windows
from features.normalization import WindowScaler
from models.pca_embedding import PCAEmbedder
from models.anomaly_detector import EmbeddingAnomalyDetector

df = load_trace(Path(SETTINGS['paths']['processed_data_dir']) / 'clean_trace.csv')
windows, _, _ = make_windows(df)
scaler = WindowScaler().fit(windows); scaler.save()
scaled = scaler.transform(windows)
emb = PCAEmbedder().fit(scaled); emb.save()
z = emb.transform(scaled)
EmbeddingAnomalyDetector().fit(z).save()
"

# 4. (Optional) Train the deep models instead of / in addition to PCA
python -m training.train_autoencoder
python -m training.train_classifier
python -m training.train_predictor

# 5. Run the API
uvicorn api.main:app --reload --port 8000
```

Then:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/dashboard/synthetic
curl http://localhost:8000/explain/gpu-000
```

## Switching between PCA and Autoencoder embeddings

By default `inference.pipeline.ClusterHeartbeatPipeline` uses the PCA
embedder (fast, no GPU needed, easy to interpret). To use the trained
autoencoder's latent space instead:

```python
pipeline = ClusterHeartbeatPipeline(use_autoencoder=True)
```

This requires `checkpoints/autoencoder.pt` to exist (run
`python -m training.train_autoencoder` first).

## Real data sources

- **Alibaba Cluster Trace / Google Borg Trace**: map their columns onto
  `utils.schema.ALL_COLUMNS` and load via `data.ingestion.FileIngestor`.
- **Live Prometheus + DCGM exporter**: use `data.ingestion.PrometheusIngestor`,
  which queries the standard `DCGM_FI_DEV_*` metric names via the Prometheus
  HTTP range-query API. Configure `data.prometheus.base_url` in `config.yaml`.

## Tests

```bash
pytest tests/ -v
```

## Notes on this build

- Storage format is CSV throughout for zero-dependency portability. If you
  have `pyarrow` installed, swap `to_csv`/`read_csv` for `to_parquet`/`read_parquet`
  in `data/synthetic_generator.py` and `data/ingestion.py` for smaller files.
- The data pipeline, PCA embedding, anomaly detection, health scoring,
  scheduling, cost optimization, and dashboard JSON were all executed
  end-to-end against a real 32-GPU synthetic trace during development.
- The PyTorch (autoencoder/classifier/LSTM) and FastAPI layers are
  implemented against their standard, documented APIs but could not be
  executed in the sandbox used to build this (no network access to
  install `torch`/`fastapi`/`shap`). Install `requirements.txt` locally
  and run `training/*.py` / `uvicorn api.main:app` to exercise them —
  flag anything that errors and it's a quick fix.
