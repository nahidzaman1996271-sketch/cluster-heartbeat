"""
Training script for Cluster Heartbeat models.
Handles training of fingerprint autoencoder and anomaly detector.
"""

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.ingestion import DataIngestion
from src.data.preprocessing import DataPreprocessor
from src.features.extractor import FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.models.fingerprint import FingerprintTrainer
from src.models.anomaly import AnomalyDetector
from src.utils.logger import get_logger
from src.config import load_config

logger = get_logger(__name__)


def train(config_path: str, output_dir: str, skip_data_gen: bool = False) -> Dict[str, Any]:
    """
    Train all models.
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for models
        skip_data_gen: Skip synthetic data generation if True
        
    Returns:
        Training results dictionary
    """
    # Load configuration
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'fingerprint_training': {},
        'anomaly_training': {},
        'data_stats': {}
    }
    
    try:
        # 1. Load data
        logger.info("="*60)
        logger.info("STEP 1: Loading Data")
        logger.info("="*60)
        
        data_ingestion = DataIngestion(config)
        df = data_ingestion.load_data(source='synthetic')
        logger.info(f"Loaded {len(df)} data points")
        results['data_stats']['total_samples'] = len(df)
        
        # 2. Preprocess data
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Preprocessing Data")
        logger.info("="*60)
        
        preprocessor = DataPreprocessor(config)
        df_clean = preprocessor.clean_data(df)
        logger.info(f"Cleaned data: {len(df_clean)} rows")
        results['data_stats']['cleaned_samples'] = len(df_clean)
        
        # 3. Extract features
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Extracting Features")
        logger.info("="*60)
        
        feature_extractor = FeatureExtractor(config)
        feature_matrix, window_features = feature_extractor.extract_all_features(df_clean)
        logger.info(f"Extracted {feature_matrix.shape[0]} windows with {feature_matrix.shape[1]} features")
        results['data_stats']['feature_dim'] = feature_matrix.shape[1] if len(feature_matrix) > 0 else 0
        results['data_stats']['num_windows'] = feature_matrix.shape[0] if len(feature_matrix) > 0 else 0
        
        if len(feature_matrix) == 0:
            logger.error("No features extracted. Training cannot proceed.")
            results['error'] = "No features extracted"
            return results
        
        # 4. Split data
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Splitting Data")
        logger.info("="*60)
        
        X_train, X_temp = train_test_split(feature_matrix, test_size=0.3, random_state=42)
        X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
        
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Validation samples: {X_val.shape[0]}")
        logger.info(f"Test samples: {X_test.shape[0]}")
        
        results['data_stats']['train_samples'] = X_train.shape[0]
        results['data_stats']['val_samples'] = X_val.shape[0]
        results['data_stats']['test_samples'] = X_test.shape[0]
        
        # 5. Normalize features
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Normalizing Features")
        logger.info("="*60)
        
        normalizer = FeatureNormalizer(config)
        X_train_norm = normalizer.fit_transform(X_train)
        X_val_norm = normalizer.transform(X_val)
        X_test_norm = normalizer.transform(X_test)
        
        logger.info(f"Features normalized using {normalizer.method} method")
        
        # Save normalizer
        normalizer_path = output_dir / 'normalizer.pkl'
        normalizer.save(str(normalizer_path))
        logger.info(f"Normalizer saved to {normalizer_path}")
        
        # 6. Train fingerprint model
        logger.info("\n" + "="*60)
        logger.info("STEP 6: Training Fingerprint Autoencoder")
        logger.info("="*60)
        
        fingerprint_trainer = FingerprintTrainer(config)
        fingerprint_trainer.build_model(X_train_norm.shape[1])
        
        train_start = time.time()
        history = fingerprint_trainer.train(X_train_norm, X_val_norm)
        train_time = time.time() - train_start
        
        logger.info(f"Fingerprint training completed in {train_time:.2f} seconds")
        
        # Save fingerprint model
        fingerprint_path = output_dir / 'fingerprint_model.pt'
        fingerprint_trainer.save_checkpoint(str(fingerprint_path))
        logger.info(f"Fingerprint model saved to {fingerprint_path}")
        
        results['fingerprint_training'] = {
            'history': history,
            'training_time': train_time,
            'best_val_loss': history.get('best_val_loss', None),
            'best_epoch': history.get('best_epoch', None),
            'model_path': str(fingerprint_path)
        }
        
        # 7. Generate fingerprints
        logger.info("\n" + "="*60)
        logger.info("STEP 7: Generating Fingerprints")
        logger.info("="*60)
        
        train_fingerprints = fingerprint_trainer.generate_fingerprints(X_train_norm)
        val_fingerprints = fingerprint_trainer.generate_fingerprints(X_val_norm)
        test_fingerprints = fingerprint_trainer.generate_fingerprints(X_test_norm)
        
        # Save fingerprints
        np.save(output_dir / 'train_fingerprints.npy', train_fingerprints)
        np.save(output_dir / 'val_fingerprints.npy', val_fingerprints)
        np.save(output_dir / 'test_fingerprints.npy', test_fingerprints)
        logger.info(f"Fingerprints saved to {output_dir}")
        
        logger.info(f"Train fingerprints shape: {train_fingerprints.shape}")
        logger.info(f"Val fingerprints shape: {val_fingerprints.shape}")
        logger.info(f"Test fingerprints shape: {test_fingerprints.shape}")
        
        # 8. Train anomaly detector
        logger.info("\n" + "="*60)
        logger.info("STEP 8: Training Anomaly Detector")
        logger.info("="*60)
        
        # Compute reconstruction errors
        recon_errors_train = fingerprint_trainer.compute_reconstruction_error(X_train_norm)
        recon_errors_val = fingerprint_trainer.compute_reconstruction_error(X_val_norm)
        recon_errors_test = fingerprint_trainer.compute_reconstruction_error(X_test_norm)
        
        anomaly_detector = AnomalyDetector(config)
        anomaly_detector.fit(train_fingerprints, recon_errors_train)
        
        # Save anomaly detector
        anomaly_path = output_dir / 'anomaly_detector.pkl'
        anomaly_detector.save(str(anomaly_path))
        logger.info(f"Anomaly detector saved to {anomaly_path}")
        
        results['anomaly_training'] = {
            'threshold': anomaly_detector.threshold,
            'model_path': str(anomaly_path)
        }
        
        # 9. Evaluate on test set
        logger.info("\n" + "="*60)
        logger.info("STEP 9: Evaluating Models")
        logger.info("="*60)
        
        # Test anomaly detection
        anomaly_results = anomaly_detector.predict(test_fingerprints, recon_errors_test)
        anomaly_rate = np.mean(anomaly_results['predictions'])
        
        logger.info(f"Anomaly rate on test set: {anomaly_rate:.4f}")
        logger.info(f"Anomaly threshold: {anomaly_detector.threshold:.4f}")
        
        # Calculate reconstruction error statistics
        recon_errors = np.linalg.norm(X_test_norm - fingerprint_trainer.reconstruct(X_test_norm), axis=1)
        logger.info(f"Reconstruction error - mean: {np.mean(recon_errors):.4f}, std: {np.std(recon_errors):.4f}")
        
        results['evaluation'] = {
            'anomaly_rate': float(anomaly_rate),
            'anomaly_threshold': float(anomaly_detector.threshold),
            'reconstruction_error_mean': float(np.mean(recon_errors)),
            'reconstruction_error_std': float(np.std(recon_errors)),
            'num_anomalies': int(np.sum(anomaly_results['predictions']))
        }
        
        # 10. Generate training report
        logger.info("\n" + "="*60)
        logger.info("STEP 10: Generating Training Report")
        logger.info("="*60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'fingerprint': {
                    'latent_dim': config['model']['fingerprint']['latent_dim'],
                    'hidden_dims': config['model']['fingerprint']['hidden_dims'],
                    'epochs': config['model']['fingerprint']['epochs'],
                    'batch_size': config['model']['fingerprint']['batch_size'],
                    'learning_rate': config['model']['fingerprint']['learning_rate']
                },
                'anomaly': {
                    'threshold_percentile': config['model']['anomaly']['threshold_percentile'],
                    'contamination': config['model']['anomaly']['contamination']
                }
            },
            'data_stats': results['data_stats'],
            'fingerprint_training': {
                'best_val_loss': results['fingerprint_training'].get('best_val_loss'),
                'best_epoch': results['fingerprint_training'].get('best_epoch'),
                'training_time': results['fingerprint_training'].get('training_time')
            },
            'anomaly_training': {
                'threshold': results['anomaly_training'].get('threshold')
            },
            'evaluation': results['evaluation'],
            'model_paths': {
                'fingerprint': str(fingerprint_path),
                'anomaly_detector': str(anomaly_path),
                'normalizer': str(normalizer_path)
            }
        }
        
        # Save report
        report_path = output_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Training report saved to {report_path}")
        
        # 11. Save final configuration
        config_path_saved = output_dir / 'config.yaml'
        with open(config_path_saved, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_path_saved}")
        
        # 12. Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"\n📊 Summary:")
        logger.info(f"  - Total samples: {results['data_stats']['total_samples']}")
        logger.info(f"  - Feature dimension: {results['data_stats']['feature_dim']}")
        logger.info(f"  - Training time: {train_time:.2f}s")
        logger.info(f"  - Best validation loss: {results['fingerprint_training'].get('best_val_loss', 'N/A'):.6f}")
        logger.info(f"  - Anomaly threshold: {results['anomaly_training'].get('threshold', 'N/A'):.4f}")
        logger.info(f"  - Test anomaly rate: {results['evaluation']['anomaly_rate']:.4f}")
        logger.info(f"\n📁 Models saved to: {output_dir}")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        results['error'] = str(e)
        return results


def evaluate_models(config_path: str, model_dir: str) -> Dict[str, Any]:
    """
    Evaluate trained models.
    
    Args:
        config_path: Path to config file
        model_dir: Directory containing trained models
        
    Returns:
        Evaluation results
    """
    logger.info("Loading models for evaluation...")
    
    config = load_config(config_path)
    model_dir = Path(model_dir)
    
    # Load data
    data_ingestion = DataIngestion(config)
    df = data_ingestion.load_data(source='synthetic')
    
    # Preprocess
    preprocessor = DataPreprocessor(config)
    df_clean = preprocessor.clean_data(df)
    
    # Extract features
    feature_extractor = FeatureExtractor(config)
    feature_matrix, _ = feature_extractor.extract_all_features(df_clean)
    
    # Load normalizer
    normalizer = FeatureNormalizer(config)
    normalizer.load(str(model_dir / 'normalizer.pkl'))
    features_norm = normalizer.transform(feature_matrix)
    
    # Load fingerprint model
    fingerprint_trainer = FingerprintTrainer(config)
    fingerprint_trainer.build_model(features_norm.shape[1])
    fingerprint_trainer.load_checkpoint(str(model_dir / 'fingerprint_model.pt'))
    
    # Generate fingerprints
    fingerprints = fingerprint_trainer.generate_fingerprints(features_norm)
    
    # Load anomaly detector
    anomaly_detector = AnomalyDetector(config)
    anomaly_detector.load(str(model_dir / 'anomaly_detector.pkl'))
    
    # Predict anomalies
    recon_errors = fingerprint_trainer.compute_reconstruction_error(features_norm)
    results = anomaly_detector.predict(fingerprints, recon_errors)
    
    return {
        'anomaly_results': results,
        'num_samples': len(features_norm),
        'fingerprints_shape': fingerprints.shape,
        'threshold': anomaly_detector.threshold
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train Cluster Heartbeat models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='models_checkpoints',
                       help='Output directory for models')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip synthetic data generation')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained models')
    
    args = parser.parse_args()
    
    if args.evaluate:
        results = evaluate_models(args.config, args.output_dir)
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2))
    else:
        train(args.config, args.output_dir, args.skip_data_gen)


if __name__ == '__main__':
    main()