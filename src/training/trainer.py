import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive trainer for MLP and CNN models on CINIC-10 dataset.

    This class handles the complete training pipeline including:
    - Training and validation loops
    - Learning rate scheduling
    - Early stopping
    - Performance monitoring
    - Model checkpointing
    - Experiment tracking

    Mathematical Foundation:
    - Loss function: CrossEntropyLoss = -Σ(y_true * log(y_pred))
    - Optimization: Adam optimizer with adaptive learning rates
    - Backpropagation: ∂L/∂w = gradient computation for weight updates
    - Learning rate scheduling: Reduces learning rate based on validation performance
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        experiment_name: str = "experiment"
    ):
        """
        Initialize the trainer.

        Args:
            model: Neural network model to train
            device: Device to train on (CPU/GPU)
            config: Training configuration dictionary
            experiment_name: Name for this experiment
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.experiment_name = experiment_name

        # Training configuration
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']

        # Initialize optimizer
        self.optimizer = self._get_optimizer()

        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize scheduler
        self.scheduler = self._get_scheduler()

        # Early stopping configuration
        early_stop_config = config['training'].get('early_stopping', {})
        self.early_stop_patience = early_stop_config.get('patience', 10)
        self.early_stop_min_delta = early_stop_config.get('min_delta', 0.001)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(f"./checkpoints/{experiment_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer based on configuration."""
        optimizer_name = self.config['training']['optimizer'].lower()

        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler based on configuration."""
        scheduler_name = self.config['training'].get('scheduler', 'none').lower()

        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Mathematical Operations:
        1. Forward pass: y_pred = model(x)
        2. Loss computation: L = CrossEntropyLoss(y_pred, y_true)
        3. Backward pass: ∂L/∂w = autograd.backward()
        4. Weight update: w = w - lr * ∂L/∂w

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Train]",
            leave=False
        )

        for batch_idx, (data, targets) in enumerate(pbar):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()

            # Update progress bar
            current_acc = 100. * correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct_predictions / total_samples

        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            pbar = tqdm(
                val_loader,
                desc=f"Epoch {self.current_epoch+1}/{self.epochs} [Val]",
                leave=False
            )

            for data, targets in pbar:
                # Move data to device
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(data)

                # Compute loss
                loss = self.criterion(outputs, targets)

                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

                # Update progress bar
                current_acc = 100. * correct_predictions / total_samples
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        # Calculate epoch metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct_predictions / total_samples

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_checkpoints: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_checkpoints: Whether to save model checkpoints

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")

        start_time = time.time()

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate epoch
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record metrics
            current_lr = self.optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)

            # Log epoch results
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                f"LR: {current_lr:.6f}"
            )

            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                if save_checkpoints:
                    self.save_checkpoint(
                        epoch,
                        is_best=True,
                        filename=f"best_model.pth"
                    )
            else:
                self.epochs_without_improvement += 1

            # Save regular checkpoint
            if save_checkpoints and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch,
                    filename=f"checkpoint_epoch_{epoch+1}.pth"
                )

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stop_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                break

        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        # Save final model and training history
        if save_checkpoints:
            self.save_checkpoint(
                self.current_epoch,
                filename="final_model.pth"
            )
            self.save_training_history()

        return self.training_history

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: str = "checkpoint.pth"
    ):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
            filename: Filename for the checkpoint
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        if is_best:
            logger.info(f"New best model saved: {filepath}")

    def load_checkpoint(self, filepath: str) -> bool:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.current_epoch = checkpoint['epoch']
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_history = checkpoint['training_history']

            logger.info(f"Checkpoint loaded from {filepath}")
            logger.info(f"Resuming from epoch {self.current_epoch}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return False

    def save_training_history(self):
        """Save training history to file."""
        history_file = self.checkpoint_dir / "training_history.yaml"

        with open(history_file, 'w') as f:
            yaml.dump(self.training_history, f)

        logger.info(f"Training history saved to {history_file}")

    def plot_training_history(self, save_plot: bool = True):
        """
        Plot training history.

        Args:
            save_plot: Whether to save the plot to file
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # Plot loss
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        # Plot learning rate
        ax3.plot(epochs, self.training_history['learning_rates'], 'g-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)

        # Plot loss difference
        loss_diff = np.array(self.training_history['val_loss']) - np.array(self.training_history['train_loss'])
        ax4.plot(epochs, loss_diff, 'purple', label='Val Loss - Train Loss')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('Overfitting Indicator')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        if save_plot:
            plot_file = self.checkpoint_dir / "training_history.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Training plot saved to {plot_file}")

        plt.show()

    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary.

        Returns:
            Dictionary containing training summary
        """
        return {
            'experiment_name': self.experiment_name,
            'model_type': self.model.__class__.__name__,
            'total_epochs': len(self.training_history['train_loss']),
            'best_val_accuracy': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'final_train_accuracy': self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0,
            'final_val_accuracy': self.training_history['val_acc'][-1] if self.training_history['val_acc'] else 0,
            'model_parameters': self.model.count_parameters() if hasattr(self.model, 'count_parameters') else 0,
            'config': self.config
        }