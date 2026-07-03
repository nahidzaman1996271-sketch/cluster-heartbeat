# load_checkpoints.py
"""
Script to load and verify checkpoint files.
"""

import torch
import pickle
import yaml
import json
from pathlib import Path
import numpy as np

def load_fingerprint_model(checkpoint_dir="models_checkpoints"):
    """Load fingerprint model checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / "fingerprint_model.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "="*60)
    print("FINGERPRINT MODEL CHECKPOINT")
    print("="*60)
    print(f"Input dimension: {checkpoint.get('input_dim', 'N/A')}")
    print(f"Latent dimension: {checkpoint.get('latent_dim', 'N/A')}")
    print(f"Hidden dims: {checkpoint.get('hidden_dims', 'N/A')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
    print(f"Training time: {checkpoint.get('training_time', 'N/A')}s")
    print(f"Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    return checkpoint

def load_anomaly_detector(checkpoint_dir="models_checkpoints"):
    """Load anomaly detector checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / "anomaly_detector.pkl"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    with open(checkpoint_path, 'rb') as f:
        detector = pickle.load(f)
    
    print("\n" + "="*60)
    print("ANOMALY DETECTOR CHECKPOINT")
    print("="*60)
    print(f"Threshold: {detector.get('threshold', 'N/A')}")
    print(f"Is fitted: {detector.get('is_fitted', 'N/A')}")
    print(f"Contamination: {detector.get('config', {}).get('contamination', 'N/A')}")
    print(f"Timestamp: {detector.get('timestamp', 'N/A')}")
    
    return detector

def load_normalizer(checkpoint_dir="models_checkpoints"):
    """Load normalizer checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / "normalizer.pkl"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    with open(checkpoint_path, 'rb') as f:
        normalizer = pickle.load(f)
    
    print("\n" + "="*60)
    print("NORMALIZER CHECKPOINT")
    print("="*60)
    print(f"Method: {normalizer.get('method', 'N/A')}")
    print(f"Is fitted: {normalizer.get('is_fitted', 'N/A')}")
    print(f"Number of features: {len(normalizer.get('feature_means', []))}")
    print(f"Timestamp: {normalizer.get('timestamp', 'N/A')}")
    
    return normalizer

def load_config(checkpoint_dir="models_checkpoints"):
    """Load training configuration."""
    config_path = Path(checkpoint_dir) / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Project: {config.get('project', {}).get('name', 'N/A')}")
    print(f"Version: {config.get('project', {}).get('version', 'N/A')}")
    print(f"Latent dim: {config.get('model', {}).get('fingerprint', {}).get('latent_dim', 'N/A')}")
    print(f"Epochs: {config.get('model', {}).get('fingerprint', {}).get('epochs', 'N/A')}")
    
    return config

def load_checkpoint_info(checkpoint_dir="models_checkpoints"):
    """Load checkpoint info."""
    info_path = Path(checkpoint_dir) / "checkpoint_info.json"
    
    if not info_path.exists():
        print(f"❌ Info not found: {info_path}")
        return None
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    print("\n" + "="*60)
    print("CHECKPOINT INFO")
    print("="*60)
    print(f"Version: {info.get('version', 'N/A')}")
    print(f"Created: {info.get('created', 'N/A')}")
    print("\nFiles:")
    for name, details in info.get('files', {}).items():
        print(f"  - {name}: {details.get('size_mb', 0):.2f} MB")
    
    return info

def verify_all_checkpoints(checkpoint_dir="models_checkpoints"):
    """Verify all checkpoint files."""
    print("\n" + "="*60)
    print("VERIFYING CHECKPOINTS")
    print("="*60)
    
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Directory not found: {checkpoint_dir}")
        return False
    
    # Check each file
    files = {
        'fingerprint_model.pt': load_fingerprint_model,
        'anomaly_detector.pkl': load_anomaly_detector,
        'normalizer.pkl': load_normalizer,
        'config.yaml': load_config,
        'checkpoint_info.json': load_checkpoint_info
    }
    
    all_loaded = True
    for filename, loader in files.items():
        filepath = checkpoint_dir / filename
        if filepath.exists():
            print(f"\n✅ {filename} exists ({filepath.stat().st_size / 1024:.1f} KB)")
            loader(checkpoint_dir)
        else:
            print(f"\n❌ {filename} missing")
            all_loaded = False
    
    return all_loaded

if __name__ == "__main__":
    verify_all_checkpoints()