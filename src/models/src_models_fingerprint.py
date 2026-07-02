"""
Fingerprint generation model for Cluster Heartbeat.
Implements autoencoder for learning compact workload representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


class FingerprintAutoencoder(nn.Module):
    """
    Autoencoder for generating workload fingerprints.
    Learns a compact representation of cluster behavior.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 32, 
                 hidden_dims: List[int] = [64, 128, 64],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 batch_norm: bool = True):
        """
        Initialize fingerprint autoencoder.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Get activation function
        if activation == 'relu':
            self.act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.act_fn = nn.ELU()
        elif activation == 'tanh':
            self.act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.ReLU()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(self.act_fn)
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(self.act_fn)
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"Built FingerprintAutoencoder: input_dim={input_dim}, latent_dim={latent_dim}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstructed input, latent representation)
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed input.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def get_latent_size(self) -> int:
        """Get the size of the latent space."""
        return self.latent_dim
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class FingerprintTrainer:
    """
    Trainer for fingerprint autoencoder model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fingerprint trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['model']['fingerprint']
        
        # Model parameters
        self.latent_dim = self.model_config.get('latent_dim', 32)
        self.hidden_dims = self.model_config.get('hidden_dims', [64, 128, 64])
        self.activation = self.model_config.get('activation', 'relu')
        self.dropout_rate = self.model_config.get('dropout_rate', 0.2)
        self.batch_norm = self.model_config.get('batch_norm', True)
        
        # Training parameters
        self.learning_rate = self.model_config.get('learning_rate', 0.001)
        self.optimizer_name = self.model_config.get('optimizer', 'adam')
        self.weight_decay = self.model_config.get('weight_decay', 0.00001)
        self.batch_size = self.model_config.get('batch_size', 64)
        self.epochs = self.model_config.get('epochs', 100)
        self.early_stopping_patience = self.model_config.get('early_stopping_patience', 10)
        self.early_stopping_min_delta = self.model_config.get('early_stopping_min_delta', 0.0001)
        self.validation_frequency = self.model_config.get('validation_frequency', 1)
        
        # Loss configuration
        self.loss_config = self.model_config.get('loss', {})
        self.reconstruction_loss = self.loss_config.get('reconstruction', 'mse')
        self.kl_weight = self.loss_config.get('kl_weight', 0.001)
        
        # Checkpoint configuration
        self.checkpoint_dir = Path(self.model_config.get('checkpoint_dir', 'models_checkpoints'))
        self.save_best_only = self.model_config.get('save_best_only', True)
        self.save_frequency = self.model_config.get('save_frequency', 10)
        
        # Model state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"FingerprintTrainer initialized on device: {self.device}")
    
    def build_model(self, input_dim: int) -> 'FingerprintTrainer':
        """
        Build the autoencoder model.
        
        Args:
            input_dim: Input feature dimension
            
        Returns:
            Self
        """
        self.model = FingerprintAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm
        ).to(self.device)
        
        # Set up optimizer
        if self.optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Set up loss function
        if self.reconstruction_loss == 'mse':
            self.criterion = nn.MSELoss()
        elif self.reconstruction_loss == 'mae':
            self.criterion = nn.L1Loss()
        elif self.reconstruction_loss == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            self.criterion = nn.MSELoss()
        
        logger.info(f"Built model with {self.model.get_parameter_count():,} parameters")
        return self
    
    def train(self, train_features: np.ndarray, 
              val_features: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
        """
        Train the autoencoder model.
        
        Args:
            train_features: Training features
            val_features: Validation features
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Convert to PyTorch tensors
        train_tensor = torch.FloatTensor(train_features).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Validation loader
        val_loader = None
        if val_features is not None:
            val_tensor = torch.FloatTensor(val_features).to(self.device)
            val_dataset = TensorDataset(val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=True
            )
        
        # Training loop
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{self.epochs}', leave=False):
                x = batch[0]
                self.optimizer.zero_grad()
                reconstructed, latent = self.model(x)
                loss = self.criterion(reconstructed, x)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None and epoch % self.validation_frequency == 0:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0]
                        reconstructed, latent = self.model(x)
                        loss = self.criterion(reconstructed, x)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                self.val_losses.append(avg_val_loss)
                
                # Update learning rate
                self.scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = avg_val_loss
                    self.best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    if self.save_best_only:
                        self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}")
            
            # Save checkpoint periodically
            if not self.save_best_only and epoch % self.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Load best model
        if self.save_best_only:
            self.load_checkpoint('best_model.pt')
        
        # Return history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'training_time': training_time
        }
    
    def generate_fingerprints(self, features: np.ndarray) -> np.ndarray:
        """
        Generate fingerprints for input features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Fingerprints of shape (n_samples, latent_dim)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            _, fingerprints = self.model(x)
        return fingerprints.cpu().numpy()
    
    def reconstruct(self, features: np.ndarray) -> np.ndarray:
        """
        Reconstruct features using the autoencoder.
        
        Args:
            features: Input features
            
        Returns:
            Reconstructed features
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            reconstructed, _ = self.model(x)
        return reconstructed.cpu().numpy()
    
    def compute_reconstruction_error(self, features: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for features.
        
        Args:
            features: Input features
            
        Returns:
            Reconstruction errors
        """
        reconstructed = self.reconstruct(features)
        errors = np.linalg.norm(features - reconstructed, axis=1)
        return errors
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        path = Path(self.checkpoint_dir) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'hidden_dims': self.model.hidden_dims
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> 'FingerprintTrainer':
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Self
        """
        path = Path(self.checkpoint_dir) / path
        
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return self
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(checkpoint['input_dim'])
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Best val loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        
        return self
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Model summary dictionary
        """
        if self.model is None:
            return {}
        
        return {
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'hidden_dims': self.model.hidden_dims,
            'parameters': self.model.get_parameter_count(),
            'device': str(self.device),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


# Convenience function
def create_fingerprint_model(config: Dict[str, Any]) -> FingerprintTrainer:
    """
    Quick function to create fingerprint model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        FingerprintTrainer instance
    """
    return FingerprintTrainer(config)