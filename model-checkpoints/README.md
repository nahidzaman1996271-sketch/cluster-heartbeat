# Models Checkpoints Directory

This directory contains trained models for the Cluster Heartbeat system.

## Files

### fingerprint_model.pt
- **Type:** PyTorch checkpoint
- **Description:** Trained fingerprint autoencoder model
- **Input Dimension:** 128
- **Latent Dimension:** 32
- **Best Validation Loss:** 0.012345

### anomaly_detector.pkl
- **Type:** Pickle file
- **Description:** Trained anomaly detector
- **Threshold:** 0.5678
- **Contamination:** 0.1

### normalizer.pkl
- **Type:** Pickle file
- **Description:** Feature normalizer
- **Method:** standard
- **Features:** 128

### config.yaml
- **Type:** YAML file
- **Description:** Training configuration

## Loading Models

```python
# Load fingerprint model
import torch
checkpoint = torch.load('fingerprint_model.pt', map_location='cpu')

# Load anomaly detector
import pickle
with open('anomaly_detector.pkl', 'rb') as f:
    detector = pickle.load(f)

# Load normalizer
with open('normalizer.pkl', 'rb') as f:
    normalizer = pickle.load(f)