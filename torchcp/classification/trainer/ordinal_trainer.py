import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Union, List
import logging
import time
from tqdm import tqdm
import numpy as np


import torch
import torch.nn as nn

from .base_trainer import Trainer


class OrdinalClassifier(nn.Module):
    """
    Method: Ordinal Classifier 
    Paper: Conformal Prediction Sets for Ordinal Classification (DEY et al., 2023)
    Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html

    Args:
        classifier (nn.Module): A PyTorch classifier model.
        phi (str, optional): The transformation function for the classifier output. Default is "abs". Options are "abs" and "square".
        varphi (str, optional): The transformation function for the cumulative sum. Default is "abs". Options are "abs" and "square".

    Attributes:
        classifier (nn.Module): The classifier model.
        phi (str): The transformation function for the classifier output.
        varphi (str): The transformation function for the cumulative sum.
        phi_function (callable): The transformation function applied to the classifier output.
        varphi_function (callable): The transformation function applied to the cumulative sum.

    Methods:
        forward(x):
            Forward pass of the model.
    
    Examples::
        >>> classifier = nn.Linear(10, 5)
        >>> model = OrdinalClassifier(classifier, phi="abs", varphi="abs")
        >>> x = torch.randn(3, 10)
        >>> output = model(x)
        >>> print(output)
    """
    
    def __init__(self, classifier, phi="abs", varphi="abs"):
        super().__init__()
        self.classifier = classifier
        self.phi = phi
        self.varphi = varphi

        phi_options = {"abs": torch.abs, "square": torch.square}
        varphi_options = {"abs": lambda x: -torch.abs(x), "square": lambda x: -torch.square(x)}

        if phi not in phi_options:
            raise NotImplementedError(f"phi function '{self.phi}' is not implemented. Options are 'abs' and 'square'.")
        if varphi not in varphi_options:
            raise NotImplementedError(f"varphi function '{self.varphi}' is not implemented. Options are 'abs' and 'square'.")
        
        self.phi_function = phi_options[phi]
        self.varphi_function = varphi_options[phi]

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the classifier, phi_function, and varphi_function.
        """
        if x.shape[1] <= 2:
            raise ValueError("The input dimension must be greater than 2.")

        x = self.classifier(x)

        # a cumulative summation
        x = torch.cat((x[:, :1], self.phi_function(x[:, 1:])), dim=1)

        x = torch.cumsum(x, dim=1)

        # the unimodal distribution
        x = self.varphi_function(x)
        return x


 
class OrdinalTrainer(Trainer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[torch.nn.Module, Callable, List[Callable]],
        device: torch.device,
        verbose: bool = True,
        loss_weights: Optional[List[float]] = None,
        ordinal_config: Dict[str, Any] = {"phi": "abs", "varphi": "abs"}
    ):
        """
        Initialize the trainer
        
        Args:
            model: The backbone model to be trained
            optimizer: Optimization algorithm
            loss_fn: Loss function(s). Can be:
                    - A PyTorch loss module
                    - A custom loss function
                    - A list of loss functions for multi-loss training
            device: Training device (CPU/GPU)
            verbose: Whether to print training progress and logs
            loss_weights: Weights for multiple loss functions if loss_fn is a list
        """
        self.backbone_model = model
        self.model = OrdinalClassifier(model, **ordinal_config)
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
                    log_msg = f"Epoch {epoch+1}/{num_epochs}"
                    for key, value in {**train_metrics, **val_metrics}.items():
                        log_msg += f" - {key}: {value:.4f}"
                    log_msg += f" - Time: {time.time() - start_time:.2f}s"
                    self.logger.info(log_msg)
            else:
                if self.verbose:
                    log_msg = f"Epoch {epoch+1}/{num_epochs}"
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