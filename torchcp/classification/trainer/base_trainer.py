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

from abc import ABC, abstractmethod
        
class BaseTrainer(ABC):
    """
    Abstract base trainer class that handles basic model setup and device configuration.
    
    Args:
        model (torch.nn.Module): Neural network model to be trained
        device (torch.device, optional): Device to run the model on. If None, will automatically use GPU ('cuda') if available, otherwise CPU ('cpu')
            Default: None
        verbose (bool): Whether to show training progress
            Default: True
    """
    
    def __init__(
            self,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True,
    ):
        if model is None:
            raise ValueError("Model cannot be None")
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = model.to(self.device)
        self.verbose = verbose
        
        # Logger setup
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def train(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            **kwargs
    ) -> torch.nn.Module:
        """
        Train the model.
        Must be implemented by subclasses.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            **kwargs: Additional training arguments
            
        Returns:
            torch.nn.Module: Trained model
        """
        pass
    
    def save_model(self, path: str) -> None:
        """
        Save model state dict to disk.
        
        Args:
            path: Path to save model weights
        """
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str) -> None:
        """
        Load model state dict from disk.
        
        Args:
            path: Path to saved model weights
        """
        self.model.load_state_dict(torch.load(path))
        
        
class Trainer(BaseTrainer):
    def __init__(self, model, device = None, verbose = True):
        super().__init__(model, device, verbose)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.CrossEntropyLoss()
            
    def calculate_loss(self, output, target):
        """
        Calculate loss using multiple loss functions
        
        Args:
            output: Model output
            target: Ground truth
            
        Returns:
            Total loss value
        """
        return self.loss_fn(output, target)  
    
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
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)



