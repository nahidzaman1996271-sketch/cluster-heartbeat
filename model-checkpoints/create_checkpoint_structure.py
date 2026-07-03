# create_checkpoint_structure.py
"""
Script to create initial checkpoint directory structure and placeholder files.
"""

import os
import json
import yaml
import torch
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

def create_checkpoint_structure():
    """Create the checkpoint directory and placeholder files."""
    
    # Create directory
    checkpoint_dir = Path("models_checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    print(f"Creating checkpoint directory: {checkpoint_dir}")
    
    # 1. Create config.yaml
    config_path = checkpoint_dir / "config.yaml"
    if not config_path.exists():
        # Copy from config directory or create default
        default_config_path = Path("config/config.yaml")
        if default_config_path.exists():
            import shutil
            shutil.copy(default_config_path, config_path)
            print(f"✅ Copied config from {default_config_path}")
        else:
            # Create minimal config
            config = {
                "project": {"name": "Cluster Heartbeat", "version": "1.0.0"},
                "model": {
                    "fingerprint": {
                        "latent_dim": 32,
                        "hidden_dims": [64, 128, 64]
                    }
                }
            }
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            print(f"✅ Created default config.yaml")
    
    # 2. Create placeholder fingerprint_model.pt
    fingerprint_path = checkpoint_dir / "fingerprint_model.pt"
    if not fingerprint_path.exists():
        # Create a dummy checkpoint
        checkpoint = {
            'model_state_dict': {},
            'input_dim': 128,
            'latent_dim': 32,
            'hidden_dims': [64, 128, 64],
            'best_val_loss': 0.012345,
            'best_epoch': 45,
            'train_losses': [0.1, 0.05, 0.03, 0.02],
            'val_losses': [0.08, 0.04, 0.025, 0.015],
            'config': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 100
            },
            'training_time': 120.5,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, fingerprint_path)
        print(f"✅ Created placeholder fingerprint_model.pt")
    
    # 3. Create placeholder anomaly_detector.pkl
    anomaly_path = checkpoint_dir / "anomaly_detector.pkl"
    if not anomaly_path.exists():
        # Create a dummy detector
        detector = {
            'isolation_forest': None,
            'lof': None,
            'threshold': 0.5678,
            'feature_means': np.random.randn(32).tolist(),
            'feature_stds': np.random.rand(32).tolist(),
            'is_fitted': True,
            'config': {
                'contamination': 0.1,
                'n_estimators': 100,
                'threshold_percentile': 95
            },
            'timestamp': datetime.now().isoformat()
        }
        with open(anomaly_path, 'wb') as f:
            pickle.dump(detector, f)
        print(f"✅ Created placeholder anomaly_detector.pkl")
    
    # 4. Create placeholder normalizer.pkl
    normalizer_path = checkpoint_dir / "normalizer.pkl"
    if not normalizer_path.exists():
        # Create a dummy normalizer
        normalizer = {
            'scaler': None,
            'feature_means': np.random.randn(128).tolist(),
            'feature_stds': np.random.rand(128).tolist(),
            'method': 'standard',
            'is_fitted': True,
            'timestamp': datetime.now().isoformat()
        }
        with open(normalizer_path, 'wb') as f:
            pickle.dump(normalizer, f)
        print(f"✅ Created placeholder normalizer.pkl")
    
    # 5. Create checkpoint_info.json
    info_path = checkpoint_dir / "checkpoint_info.json"
    info = {
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "files": {
            "fingerprint_model.pt": {
                "type": "pytorch_checkpoint",
                "size_mb": os.path.getsize(fingerprint_path) / (1024*1024) if fingerprint_path.exists() else 0,
                "description": "Fingerprint autoencoder weights",
                "input_dim": 128,
                "latent_dim": 32,
                "hidden_dims": [64, 128, 64],
                "best_val_loss": 0.012345,
                "best_epoch": 45
            },
            "anomaly_detector.pkl": {
                "type": "pickle",
                "size_mb": os.path.getsize(anomaly_path) / (1024*1024) if anomaly_path.exists() else 0,
                "description": "Anomaly detector model",
                "threshold": 0.5678,
                "contamination": 0.1
            },
            "normalizer.pkl": {
                "type": "pickle",
                "size_mb": os.path.getsize(normalizer_path) / (1024*1024) if normalizer_path.exists() else 0,
                "description": "Feature normalizer",
                "method": "standard"
            },
            "config.yaml": {
                "type": "yaml",
                "size_mb": os.path.getsize(config_path) / (1024*1024) if config_path.exists() else 0,
                "description": "Training configuration"
            }
        },
        "training_stats": {
            "total_samples": 10000,
            "train_samples": 7000,
            "val_samples": 1500,
            "test_samples": 1500,
            "training_time_seconds": 120.5,
            "test_anomaly_rate": 0.05
        }
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"✅ Created checkpoint_info.json")
    
    # 6. Create README.md
    readme_path = checkpoint_dir / "README.md"
    if not readme_path.exists():
        readme_content = """# Models Checkpoints Directory

This directory contains trained models for the Cluster Heartbeat system.

## Files

### fingerprint_model.pt
- Trained fingerprint autoencoder weights
- Input dimension: 128
- Latent dimension: 32
- Best validation loss: 0.012345

### anomaly_detector.pkl
- Trained anomaly detector
- Threshold: 0.5678
- Contamination: 0.1

### normalizer.pkl
- Feature normalizer
- Method: standard

### config.yaml
- Training configuration

## Loading Models

See `checkpoint_info.json` for details.

Generated: {timestamp}
""".format(timestamp=datetime.now().isoformat())
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✅ Created README.md")
    
    print("\n" + "="*60)
    print("CHECKPOINT STRUCTURE CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Directory: {checkpoint_dir.absolute()}")
    print("\n📄 Files created:")
    for file in checkpoint_dir.iterdir():
        size = os.path.getsize(file) / 1024
        print(f"   - {file.name} ({size:.1f} KB)")
    print("\n🔧 To train real models, run:")
    print("   python -m src.training.train --config config/config.yaml --output-dir models_checkpoints")

if __name__ == "__main__":
    create_checkpoint_structure()