# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchcp.classification.loss import UncertaintyAwareLoss
from torchcp.classification.trainer import Trainer


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
        criterion_pred_loss_fn (torch.nn.Module): Loss function for accuracy.
        mu (float): A hyperparameter controlling the weight of the conformal loss term in the total loss (default: 0.2).
        alpha (float): A significance level for the conformal loss function (default: 0.1).
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
                 optimizer: torch.optim.Optimizer,
                 loss_fn=torch.nn.CrossEntropyLoss(),
                 loss_weights=None,
                 device: torch.device=None,
                 verbose: bool = True,
                 mu: float = 0.2,
                 alpha: float = 0.1):
        super(UncertaintyAwareTrainer, self).__init__(model, optimizer, loss_fn, loss_weights, device, verbose)

        self.conformal_loss_fn = UncertaintyAwareLoss()
        self.mu = mu
        self.alpha = alpha

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
        if training:
            idx_ce = torch.where(Z_batch == 0)[0]
            loss_ce = self.loss_fn(output[idx_ce], target[idx_ce])
        else:
            Z_batch = torch.ones(len(output)).long().to(self.device)
            loss_ce = self.loss_fn(output, target)

        loss_scores = self.conformal_loss_fn(output, target, Z_batch)

        loss = loss_ce + loss_scores * self.mu
        return loss

    def train_epoch(self, train_loader: DataLoader):
        """
        Trains the model for one epoch.

        The function iterates through the training data and updates the model parameters
        using backpropagation and the optimizer.

        Args:
            train_loader (torch.utils.data.DataLoader): The DataLoader providing the training data.
        """

        self.model.train()

        for X_batch, Y_batch, Z_batch in train_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            Z_batch = Z_batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(X_batch)

            # Calculate loss
            loss = self.calculate_loss(output, Y_batch, Z_batch)

            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        """
        Validates the model on the validation set.

        The function computes both the loss and accuracy for the validation data.

        Args:
            val_loader (torch.utils.data.DataLoader): The DataLoader providing the validation data.

        Returns:
            tuple: The average loss and accuracy for the validation set.
        """
        loss_val = 0
        acc_val = 0

        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            output = self.model(X_batch)
            loss = self.calculate_loss(output, Y_batch, None, training=False)
            pred = output.argmax(dim=1)
            acc = pred.eq(Y_batch).sum()

            loss_val += loss.item()
            acc_val += acc.item()

        metrics = {
            'val_loss': loss_val / len(val_loader),
            'val_acc': acc_val / len(val_loader)
        }

        return metrics

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader = None,
              num_epochs: int = 10,
              save_path: str = None):
        """
        Trains the model for multiple epochs and optionally performs early stopping
        based on validation loss or accuracy. Saves the best models during training.

        Args:
            train_loader (torch.utils.data.DataLoader): The DataLoader for the training set.
            save_path (str): The path where model checkpoints should be saved.
            val_loader (torch.utils.data.DataLoader): The DataLoader for the validation set (default: None).
            num_epochs (int): The number of epochs to train for (default: 10).
        """

        train_loader = self.split_dataloader(train_loader)
        lr_milestones = [int(num_epochs * 0.5)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=0.1)

        if self.verbose:
            epoch_iter = tqdm(range(num_epochs), desc="Training")
        else:
            epoch_iter = range(num_epochs)

        best_val_acc = 0

        for epoch in epoch_iter:

            self.train_epoch(train_loader)

            scheduler.step()
            self.model.eval()

            if val_loader is not None:
                val_metrics = self.validate(val_loader)

                if val_metrics['val_acc'] > best_val_acc and save_path:
                    best_val_acc = val_metrics['val_acc']
                    self.save_checkpoint(epoch, save_path, val_metrics)

                    if self.verbose:
                        self.logger.info(f"Saved best model with validation accuracy: {val_metrics['val_acc']:.4f}")


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
