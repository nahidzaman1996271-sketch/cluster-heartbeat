"""
Main entry point for Cluster Heartbeat.
Provides CLI interface and orchestration of all components.
"""

import sys
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from .config import load_config
from .core.service import ClusterHeartbeatService
from .utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def run_api(config_path: Optional[str] = None):
    """
    Run the API server.
    
    Args:
        config_path: Path to configuration file
    """
    import uvicorn
    
    # Load configuration
    config = load_config(config_path)
    
    logger.info(f"Starting Cluster Heartbeat API on {config['api']['host']}:{config['api']['port']}")
    logger.info(f"Environment: {config['project']['environment']}")
    logger.info(f"Debug mode: {config['project']['debug']}")
    
    # Initialize service
    service = ClusterHeartbeatService(config)
    service.start()
    
    # Run API
    try:
        uvicorn.run(
            "src.api.main:app",
            host=config['api']['host'],
            port=config['api']['port'],
            workers=config['api']['workers'],
            reload=config['api']['reload'],
            log_level=config['logging']['level'].lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Shutting down API server...")
    finally:
        service.stop()


def run_training(config_path: Optional[str] = None, output_dir: Optional[str] = None):
    """
    Run model training.
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory to save models
    """
    from .training.train import train
    
    logger.info("Starting model training...")
    
    if output_dir is None:
        output_dir = "models_checkpoints"
    
    train(config_path, output_dir)
    logger.info("Training completed!")


def run_test(config_path: Optional[str] = None):
    """
    Run system test.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("Running system test...")
    
    config = load_config(config_path)
    
    # Initialize service
    service = ClusterHeartbeatService(config)
    service.start()
    
    # Generate test data
    from .data.ingestion import DataIngestion
    ingestion = DataIngestion(config)
    df = ingestion.load_data('synthetic')
    
    logger.info(f"Generated {len(df)} test data points")
    
    # Process data
    results = service.process_metrics(df.to_dict('records'))
    
    # Print results
    print("\n" + "="*60)
    print("CLUSTER HEARTBEAT - TEST RESULTS")
    print("="*60)
    
    summary = results.get('summary', {})
    print(f"\n📊 Cluster Status:")
    print(f"  - Status: {summary.get('health_status', 'unknown')}")
    print(f"  - Total Nodes: {summary.get('total_nodes', 0)}")
    print(f"  - Average Health: {summary.get('average_health', 0):.2f}%")
    print(f"  - Anomalies Detected: {summary.get('anomaly_count', 0)}")
    print(f"  - Idle GPUs: {summary.get('idle_gpus', 0)}")
    print(f"  - Potential Savings: ${summary.get('cost_savings', 0):.2f}")
    
    print(f"\n📈 Health Scores:")
    health_scores = results.get('health_scores', {})
    node_scores = health_scores.get('node_scores', {})
    for node_id, score in node_scores.items():
        status = "🟢 Good" if score > 70 else "🟡 Warning" if score > 50 else "🔴 Critical"
        print(f"  - Node {node_id}: {score:.2f}% ({status})")
    
    # Show sample predictions
    print(f"\n🔮 Sample Predictions:")
    anomaly_results = results.get('anomaly_results', {})
    scores = anomaly_results.get('scores', [])
    if scores:
        for i, score in enumerate(scores[:5]):
            print(f"  - Job {i}: Anomaly Score = {score:.3f}")
    
    print("\n" + "="*60)
    print("✅ System test completed successfully!")
    print("="*60)
    
    service.stop()
    
    return results


def run_cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Cluster Heartbeat - AI-powered GPU cluster intelligence system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode api                # Start API server
  %(prog)s --mode train              # Train models
  %(prog)s --mode test               # Run system test
  %(prog)s --mode api --config custom_config.yaml  # Use custom config
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['api', 'train', 'test', 'explore'],
        default='api',
        help='Run mode: api, train, test, explore (default: api)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models_checkpoints',
        help='Output directory for trained models (default: models_checkpoints)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='API port (overrides config)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        help='API host (overrides config)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        os.environ['DEBUG'] = 'true'
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Override config with command line args
    if args.port or args.host:
        config = load_config(args.config)
        if args.port:
            config['api']['port'] = args.port
        if args.host:
            config['api']['host'] = args.host
    
    # Execute based on mode
    try:
        if args.mode == 'api':
            run_api(args.config)
        elif args.mode == 'train':
            run_training(args.config, args.output_dir)
        elif args.mode == 'test':
            run_test(args.config)
        elif args.mode == 'explore':
            run_exploration(args.config)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


def run_exploration(config_path: Optional[str] = None):
    """
    Run data exploration mode.
    
    Args:
        config_path: Path to configuration file
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    logger.info("Starting data exploration mode...")
    
    config = load_config(config_path)
    
    # Load data
    from .data.ingestion import DataIngestion
    ingestion = DataIngestion(config)
    df = ingestion.load_data('synthetic')
    
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\n📊 Data Overview:")
    print(f"  - Total records: {len(df)}")
    print(f"  - Columns: {', '.join(df.columns)}")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n📈 Statistics:")
    print(df.describe())
    
    print(f"\n🔍 Sample Data:")
    print(df.head())
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"\n⚠️  Null Values Detected:")
        print(null_counts[null_counts > 0])
    
    # Create visualization
    if len(df.columns) > 1:
        print("\n📈 Generating visualization...")
        
        # Plot first few columns
        cols = [col for col in df.columns if col not in ['timestamp', 'node_id', 'job_id']][:6]
        if cols:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            for i, col in enumerate(cols):
                if i < len(axes):
                    df[col].plot(ax=axes[i], title=col)
                    axes[i].set_xlabel('Time')
                    axes[i].set_ylabel(col)
            
            # Hide empty subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to data_exploration.png")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    run_cli()