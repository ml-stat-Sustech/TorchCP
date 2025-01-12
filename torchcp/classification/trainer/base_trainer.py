# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable, Union, List
import copy

class BaseTrainer:
    """
    Base trainer class that handles basic model setup and device configuration.
    
    Args:
        model (torch.nn.Module): Neural network model
        device (torch.device): Device to run on (CPU/GPU)
        verbose (bool): Whether to show training progress
        
    Examples:
        >>> model = MyModel()
        >>> base_trainer = BaseTrainer(model)
    """
    
    training_algorithm = None

    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = model.to(self.device)
        self.verbose = verbose
        # Setup logging
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            
    def train(self,  train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 10,):
        if self.training_algorithm is None:
            raise NotImplementedError("Training algorithm is not defined")
        self.model = self.training_algorithm.train(train_loader, val_loader, num_epochs)
        return self.model
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
            
            
            
class TrainingAlgorithm:
    """
    Implementation of specific training algorithms and procedures.
    
    This class handles the core training logic including loss calculation,
    optimization steps, and validation procedures. It supports single or multiple
    loss functions with optional weights.
    
    Attributes:
        model: PyTorch model to train
        optimizer: Optimization algorithm
        loss_fn: Single loss function or list of loss functions
        loss_weights: Weights for multiple loss functions
        device: Training device (CPU/GPU)
        verbose: Whether to show training progress
        logger: Logging instance
        
    Args:
        model (torch.nn.Module): Neural network model
        optimizer (torch.optim.Optimizer): Optimization algorithm
        loss_fn (Union[torch.nn.Module, Callable, List[Callable]]): Loss function(s)
        loss_weights (Optional[List[float]]): Weights for multiple loss functions
        device (torch.device): Training device
        verbose (bool): Whether to show training progress
        
    Examples:
        >>> # Define a model and loss functions
        >>> model = nn.Sequential(nn.Linear(10, 1))
        >>> mse_loss = nn.MSELoss()
        >>> l1_loss = nn.L1Loss()
        >>> 
        >>> # Create optimizer
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> 
        >>> # Initialize training algorithm with multiple losses
        >>> algorithm = TrainingAlgorithm(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=[mse_loss, l1_loss],
        ...     loss_weights=[0.7, 0.3],
        ...     device=torch.device('cuda'),
        ...     verbose=True
        ... )
        >>> 
        >>> # Train the model
        >>> trained_model = algorithm.train(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=10
        ... )
        >>> 
        >>> # Validate on test set
        >>> test_loss = algorithm.validate(test_loader)
    """
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: Union[torch.nn.Module, Callable, List[Callable]],
            loss_weights: Optional[List[float]] = None,
            device: torch.device = None,
            verbose: bool = True,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = model.to(self.device)
        
        self.optimizer = optimizer
        self.verbose = verbose
        
        if not isinstance(loss_fn, list):
            self.loss_fn = [loss_fn]
        else:
            self.loss_fn = loss_fn
        if loss_weights is None:
                self.loss_weights = torch.ones(len(self.loss_fn), device=self.device)
        else:
            if len(loss_weights) != len(self.loss_fn):
                raise ValueError(f"Number of loss functions must match number of weights")
            self.loss_weights = torch.tensor(loss_weights, device=self.device)
        
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
        total_loss = 0
        for fn, weight in zip(self.loss_fn, self.loss_weights):
            loss = fn(output, target)
            total_loss += weight * loss
        return total_loss

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss for this epoch
        """
        self.model.train()
        total_loss = 0

        # Create progress bar if verbose
        train_iter = tqdm(train_loader, desc="Training") if self.verbose else train_loader

        for data, target in train_iter:
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.calculate_loss(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.verbose:
                train_iter.set_postfix({'loss': loss.item()})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """
        Evaluate model on validation set
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc="Validating") if self.verbose else val_loader
            
            for data, target in val_iter:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.calculate_loss(output, target)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            num_epochs: int = 10,
    ):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of training epochs
        """
        best_loss = float('inf')
        best_model_state = None

        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(train_loader)
            log_msg = f"Epoch {epoch + 1}/{num_epochs} - train_loss: {train_loss:.4f}"

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                log_msg += f" - val_loss: {val_loss:.4f}"
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    log_msg += f"\nNew best model with val_loss: {val_loss:.4f}"

            log_msg += f" - Time: {time.time() - start_time:.2f}s"
            if self.verbose:
                self.logger.info(log_msg)

        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if self.verbose:
                self.logger.info(f"Loaded best model with val_loss: {best_loss:.4f}")

        return self.model


