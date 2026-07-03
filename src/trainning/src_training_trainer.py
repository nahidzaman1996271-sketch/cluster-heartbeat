"""
Base trainer class for Cluster Heartbeat models.
Provides common training utilities and abstractions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from pathlib import Path
import logging
import time
import json
from dataclasses import dataclass, field
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """
    Configuration for model training.
    """
    # Training parameters
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Optimizer
    optimizer: str = 'adam'  # adam, sgd, rmsprop
    scheduler: Optional[str] = 'reduce_on_plateau'  # reduce_on_plateau, cosine, step
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    
    # Checkpoint
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    save_frequency: int = 10
    
    # Device
    device: Optional[str] = None
    
    # Logging
    log_frequency: int = 10
    use_tensorboard: bool = False
    tensorboard_dir: str = 'runs'
    
    # Mixed precision
    use_amp: bool = False
    
    # Gradient clipping
    gradient_clip: Optional[float] = None
    
    def __post_init__(self):
        """Initialize device if not specified."""
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


class ModelTrainer:
    """
    Base trainer class for PyTorch models.
    Provides common training, validation, and checkpointing functionality.
    """
    
    def __init__(self, config: Dict[str, Any], model: Optional[nn.Module] = None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            model: PyTorch model to train
        """
        self.config = config
        self.model = model
        self.trainer_config = self._parse_trainer_config(config)
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.scaler = None  # For mixed precision
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        
        # Device
        self.device = torch.device(self.trainer_config.device)
        
        logger.info(f"ModelTrainer initialized on device: {self.device}")
    
    def _parse_trainer_config(self, config: Dict[str, Any]) -> TrainerConfig:
        """
        Parse trainer configuration from config dict.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            TrainerConfig object
        """
        # Get model-specific config
        model_config = config.get('model', {})
        fingerprint_config = model_config.get('fingerprint', {})
        
        return TrainerConfig(
            epochs=fingerprint_config.get('epochs', 100),
            batch_size=fingerprint_config.get('batch_size', 64),
            learning_rate=fingerprint_config.get('learning_rate', 0.001),
            weight_decay=fingerprint_config.get('weight_decay', 0.0001),
            optimizer=fingerprint_config.get('optimizer', 'adam'),
            early_stopping_patience=fingerprint_config.get('early_stopping_patience', 10),
            early_stopping_min_delta=fingerprint_config.get('early_stopping_min_delta', 0.0001),
            checkpoint_dir=fingerprint_config.get('checkpoint_dir', 'checkpoints'),
            save_best_only=fingerprint_config.get('save_best_only', True),
            save_frequency=fingerprint_config.get('save_frequency', 10),
            device=str(self.device) if hasattr(self, 'device') else None
        )
    
    def set_model(self, model: nn.Module) -> 'ModelTrainer':
        """
        Set the model to train.
        
        Args:
            model: PyTorch model
            
        Returns:
            Self
        """
        self.model = model.to(self.device)
        return self
    
    def set_criterion(self, criterion: nn.Module) -> 'ModelTrainer':
        """
        Set the loss function.
        
        Args:
            criterion: Loss function
            
        Returns:
            Self
        """
        self.criterion = criterion
        return self
    
    def set_optimizer(self, optimizer: Optional[torch.optim.Optimizer] = None) -> 'ModelTrainer':
        """
        Set or create optimizer.
        
        Args:
            optimizer: Optional optimizer instance
            
        Returns:
            Self
        """
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            # Create optimizer from config
            params = self.model.parameters()
            lr = self.trainer_config.learning_rate
            weight_decay = self.trainer_config.weight_decay
            
            if self.trainer_config.optimizer == 'adam':
                self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            elif self.trainer_config.optimizer == 'sgd':
                self.optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
            elif self.trainer_config.optimizer == 'rmsprop':
                self.optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
            else:
                self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        return self
    
    def set_scheduler(self, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> 'ModelTrainer':
        """
        Set or create learning rate scheduler.
        
        Args:
            scheduler: Optional scheduler instance
            
        Returns:
            Self
        """
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            if self.trainer_config.scheduler == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            elif self.trainer_config.scheduler == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.trainer_config.epochs
                )
            elif self.trainer_config.scheduler == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=30,
                    gamma=0.1
                )
            else:
                self.scheduler = None
        
        return self
    
    def setup_mixed_precision(self) -> 'ModelTrainer':
        """Setup mixed precision training."""
        if self.trainer_config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        return self
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.trainer_config.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            if isinstance(batch, (list, tuple)):
                batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
                x = batch[0]
                y = batch[1] if len(batch) > 1 else batch[0]
            else:
                x = batch.to(self.device)
                y = batch.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    output = self.model(x)
                    loss = self.criterion(output, y)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.trainer_config.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                
                # Gradient clipping
                if self.trainer_config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.gradient_clip)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.trainer_config.log_frequency == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                if isinstance(batch, (list, tuple)):
                    batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else batch[0]
                else:
                    x = batch.to(self.device)
                    y = batch.to(self.device)
                
                output = self.model(x)
                loss = self.criterion(output, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not set. Call set_model() first.")
        
        if self.criterion is None:
            raise ValueError("Criterion not set. Call set_criterion() first.")
        
        if self.optimizer is None:
            self.set_optimizer()
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        early_stopping_counter = 0
        
        for epoch in range(self.trainer_config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                if val_loss < self.best_val_loss - self.trainer_config.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    early_stopping_counter = 0
                    
                    # Save best model
                    if self.trainer_config.save_best_only:
                        self.best_model_state = self.model.state_dict().copy()
                        self.save_checkpoint('best_model.pt')
                    
                    logger.info(f"New best model! Val loss: {val_loss:.6f}")
                else:
                    early_stopping_counter += 1
                
                # Early stopping
                if early_stopping_counter >= self.trainer_config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.6f}")
            
            # Save checkpoint
            if not self.trainer_config.save_best_only and epoch % self.trainer_config.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        self.train_losses = history['train_loss']
        self.val_losses = history.get('val_loss', [])
        
        return history
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        checkpoint_dir = Path(self.trainer_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / path
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'trainer_config': self.trainer_config.__dict__,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, path: str) -> 'ModelTrainer':
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            
        Returns:
            Self
        """
        checkpoint_path = Path(self.trainer_config.checkpoint_dir) / path
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return self
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
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
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'trainer_config': {
                'epochs': self.trainer_config.epochs,
                'batch_size': self.trainer_config.batch_size,
                'learning_rate': self.trainer_config.learning_rate,
                'optimizer': self.trainer_config.optimizer
            }
        }


class AutoencoderTrainer(ModelTrainer):
    """
    Specialized trainer for autoencoder models.
    """
    
    def __init__(self, config: Dict[str, Any], model: Optional[nn.Module] = None):
        """
        Initialize autoencoder trainer.
        
        Args:
            config: Configuration dictionary
            model: Autoencoder model
        """
        super().__init__(config, model)
        
        # Set default criterion for autoencoders
        if self.criterion is None:
            self.set_criterion(nn.MSELoss())
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """
        Train autoencoder for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.trainer_config.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed, _ = self.model(x)
            loss = self.criterion(reconstructed, x)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.trainer_config.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.trainer_config.gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % self.trainer_config.log_frequency == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate autoencoder.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                reconstructed, _ = self.model(x)
                loss = self.criterion(reconstructed, x)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches


# Convenience functions
def create_trainer(config: Dict[str, Any], model_type: str = 'autoencoder') -> ModelTrainer:
    """
    Create a trainer instance.
    
    Args:
        config: Configuration dictionary
        model_type: Type of trainer ('autoencoder', 'classifier', 'regressor')
        
    Returns:
        ModelTrainer instance
    """
    if model_type == 'autoencoder':
        return AutoencoderTrainer(config)
    else:
        return ModelTrainer(config)