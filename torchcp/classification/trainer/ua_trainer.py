# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchcp.classification.loss import UncertaintyAwareLoss
import copy
from torchcp.classification.trainer.base_trainer import Trainer

class TrainDataset(Dataset):
    def __init__(self, X_data, Y_data, Z_data):
        self.X_data = X_data
        self.Y_data = Y_data
        self.Z_data = Z_data

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index], self.Z_data[index]

    def __len__(self):
        return len(self.X_data)

class UncertaintyAwareTrainer(Trainer):
    """
    Conformalized uncertainty-aware training of deep multi-class classifiers

    Args:
        model (torch.nn.Module): Neural network model to train.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        base_loss (torch.nn.Module): The base loss function .
        loss_weight (float): The weight of the conformal loss term in the total loss.
        device (torch.device): Device to run on (CPU/GPU) (default: CPU).

    Examples:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss_fn = nn.CrossEntropyLoss()
        >>> trainer = ConfLearnTrainer(model, optimizer, loss_fn)
        >>> save_path = './path/to/save'
        >>> trainer.train(train_loader, save_path, val_loader, num_epochs=10)

    Reference:
        Einbinder et al. "Training Uncertainty-Aware Classifiers with Conformalized Deep Learning" (2022), https://arxiv.org/abs/2205.05878
    """

    def __init__(self,
                model: torch.nn.Module,
                optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                optimizer_params: dict = {},
                base_loss=torch.nn.CrossEntropyLoss(),
                loss_weight: float = None,
                device: torch.device = None,
                verbose: bool = True):
        
        super(UncertaintyAwareTrainer, self).__init__(model, device, verbose)
        self.optimizer = optimizer_class(
            model.parameters(),
            **optimizer_params
        )

        self.conformal_loss_fn = UncertaintyAwareLoss()
        self.loss_fns = [base_loss, self.conformal_loss_fn]
        if loss_weight is None:
            loss_weight = 0.2
        self.loss_weights = [1.0, loss_weight]
    
    def train(self,  train_loader: DataLoader,
              val_loader: DataLoader = None,
              num_epochs: int = 10,):
        lr_milestones = [int(num_epochs * 0.5)]
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=0.1)
        train_loader = self.split_dataloader(train_loader)
    
        return super().train(train_loader, val_loader, num_epochs)
    
    def train_epoch(self, train_loader: DataLoader):
        """
        Trains the model for one epoch.

        The function iterates through the training data and updates the model parameters
        using backpropagation and the optimizer.

        Args:
            train_loader (torch.utils.data.DataLoader): The DataLoader providing the training data.
        """

        self.model.train()
        total_loss = 0
        
        train_iter = tqdm(train_loader, desc="Training") if self.verbose else train_loader

        for X_batch, Y_batch, Z_batch in train_iter:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            Z_batch = Z_batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(X_batch)

            # Calculate loss
            loss = self.calculate_loss(output, Y_batch, Z_batch)

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.verbose:
                train_iter.set_postfix({'loss': loss.item()})
        self.scheduler.step()

        return total_loss / len(train_loader)

    def calculate_loss(self, output, target, Z_batch, training=True):
        """
        Calculates the total loss during training or validation.

        The loss is a combination of the prediction loss and the conformal prediction loss,
        where the conformal loss is weighted by the hyperparameter `mu`.

        Args:
            output (torch.Tensor): The model's output predictions (logits).
            target (torch.Tensor): The true labels (ground truth).
            Z_batch (torch.Tensor): A tensor indicating which samples are used for conformal prediction loss.
            training (bool): A flag indicating whether the calculation is for training or validation (default: True).

        Returns:
            torch.Tensor: The computed total loss.
        """
        total_loss = 0
        if training:
            for fn, weight, loss_idx in zip(self.loss_fns, self.loss_weights, range(len(self.loss_fns))):
                idx_type = torch.where(Z_batch == loss_idx)[0]
                if loss_idx == 0:
                    loss = fn(output[idx_type], target[idx_type])
                else:
                    loss = fn(output[idx_type], target[idx_type])
                total_loss += weight * loss
            return total_loss
        else:
            for fn, weight in zip(self.loss_fns, self.loss_weights):
                loss = fn(output, target)
                total_loss += weight * loss
        return total_loss
    
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
                loss = self.calculate_loss(output, target, Z_batch=None, training=False)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def split_dataloader(self, data_loader: DataLoader, split_ratio=0.8):
        """
        This function splits a given DataLoader into two parts based on the specified split ratio
        for calculate cross-entropy loss and conformal loss, respectively.
        The split is done randomly, and the labels for the split data are generated as a binary indicator.

        Args:
            data_loader (DataLoader): The DataLoader object containing the original dataset.
            split_ratio (float, optional): The ratio to split the dataset into two parts.
        
        Returns:
            DataLoader: A new DataLoader that contains the modified dataset with the binary labels.
        """

        x_list = []
        y_list = []
        for tmp_x, tmp_y in data_loader:
            x_list.append(tmp_x)
            y_list.append(tmp_y)
        X_data = torch.cat(x_list)
        Y_data = torch.cat(y_list)

        Z_data = torch.zeros(len(X_data)).long().to(self.device)
        split = int(len(X_data) * split_ratio)
        Z_data[torch.randperm(len(X_data))[split:]] = 1

        train_dataset = TrainDataset(X_data, Y_data, Z_data)
        train_loader = DataLoader(
            train_dataset, batch_size=data_loader.batch_size, shuffle=True, drop_last=data_loader.drop_last)
        
        return train_loader
