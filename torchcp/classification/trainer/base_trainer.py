import logging
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable, Union, List


class Trainer:
    """
    A general-purpose PyTorch model trainer.
    
    Args:
        model: PyTorch model instance
        optimizer: Optimizer instance
        loss_fn: Single loss function or list of loss functions
        device: Computing device (CPU/GPU)
        verbose: Whether to show detailed logs
        loss_weights: List of weights for multiple loss functions
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: Union[torch.nn.Module, Callable, List[Callable]],
            device: torch.device,
            verbose: bool = True,
            loss_weights: Optional[List[float]] = None
    ):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose

        # Handle single or multiple loss functions
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        if isinstance(loss_fn, list):
            assert loss_weights is not None, "Must provide weights when using multiple loss functions"
            assert len(loss_fn) == len(loss_weights), "Number of loss functions must match number of weights"

        # Setup logging
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

    def calculate_loss(self, output, target):
        """
        Calculate loss using single or multiple loss functions
        
        Args:
            output: Model output
            target: Ground truth
            
        Returns:
            Total loss value
        """
        if isinstance(self.loss_fn, list):
            total_loss = 0
            for fn, weight in zip(self.loss_fn, self.loss_weights):
                loss = fn(output, target)
                total_loss += weight * loss
            return total_loss
        else:
            return self.loss_fn(output, target)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        individual_losses = {} if isinstance(self.loss_fn, list) else None

        # Create progress bar if verbose
        if self.verbose:
            train_iter = tqdm(train_loader, desc="Training")
        else:
            train_iter = train_loader

        for batch_idx, (data, target) in enumerate(train_iter):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            # Calculate loss
            loss = self.calculate_loss(output, target)

            # Track individual losses if using multiple loss functions
            if isinstance(self.loss_fn, list):
                for i, (fn, weight) in enumerate(zip(self.loss_fn, self.loss_weights)):
                    individual_loss = fn(output, target).item()
                    individual_losses[f'loss_{i}'] = individual_losses.get(f'loss_{i}', 0) + individual_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if self.verbose:
                train_iter.set_postfix({'loss': loss.item()})

        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        metrics = {'loss': avg_loss}

        if individual_losses:
            for key in individual_losses:
                metrics[key] = individual_losses[key] / len(train_loader)

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        individual_losses = {} if isinstance(self.loss_fn, list) else None

        with torch.no_grad():
            # Create progress bar if verbose
            if self.verbose:
                val_iter = tqdm(val_loader, desc="Validating")
            else:
                val_iter = val_loader

            for data, target in val_iter:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                loss = self.calculate_loss(output, target)
                total_loss += loss.item()

                # Track individual losses
                if isinstance(self.loss_fn, list):
                    for i, (fn, weight) in enumerate(zip(self.loss_fn, self.loss_weights)):
                        individual_loss = fn(output, target).item()
                        individual_losses[f'val_loss_{i}'] = individual_losses.get(f'val_loss_{i}', 0) + individual_loss

                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        # Calculate average metrics
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': correct / total
        }

        if individual_losses:
            for key in individual_losses:
                metrics[key] = individual_losses[key] / len(val_loader)

        return metrics

    def train(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            num_epochs: int = 10,
            save_path: Optional[str] = None
    ):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of training epochs
            save_path: Optional path to save the best model
        """
        best_val_acc = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

                # Save best model
                if val_metrics['val_acc'] > best_val_acc and save_path:
                    best_val_acc = val_metrics['val_acc']
                    torch.save(self.model.state_dict(), save_path)
                    if self.verbose:
                        self.logger.info(f"Saved best model with validation accuracy: {val_metrics['val_acc']:.4f}")

                if self.verbose:
                    # Construct log message
                    log_msg = f"Epoch {epoch + 1}/{num_epochs}"
                    for key, value in {**train_metrics, **val_metrics}.items():
                        log_msg += f" - {key}: {value:.4f}"
                    log_msg += f" - Time: {time.time() - start_time:.2f}s"
                    self.logger.info(log_msg)
            else:
                if self.verbose:
                    log_msg = f"Epoch {epoch + 1}/{num_epochs}"
                    for key, value in train_metrics.items():
                        log_msg += f" - {key}: {value:.4f}"
                    log_msg += f" - Time: {time.time() - start_time:.2f}s"
                    self.logger.info(log_msg)

    def save_checkpoint(
            self,
            epoch: int,
            save_path: str,
            metrics: Optional[Dict[str, float]] = None
    ):
        """
        Save a checkpoint
        
        Args:
            epoch: Current training epoch
            save_path: Path to save the checkpoint
            metrics: Optional dictionary of metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, save_path)
        if self.verbose:
            self.logger.info(f"Saved checkpoint at epoch {epoch}")

    def load_checkpoint(self, load_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint
        
        Args:
            load_path: Path to the checkpoint
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint
