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
from torchcp.classification.loss import UniformLoss


class TrainDataset(Dataset):
    def __init__(self, X_data, Y_data, Z_data):
        self.X_data = X_data
        self.Y_data = Y_data
        self.Z_data = Z_data

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index], self.Z_data[index]

    def __len__(self):
        return len(self.X_data)


class UniformTrainer:
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
    """

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion_pred_loss_fn=torch.nn.CrossEntropyLoss(),
                 mu: float = 0.2,
                 alpha: float = 0.1,
                 device: torch.device = 'cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer

        self.criterion_pred_loss_fn = criterion_pred_loss_fn
        self.conformal_loss_fn = UniformLoss()
        self.mu = mu
        self.alpha = alpha
        self.device = device

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
            loss_ce = self.criterion_pred_loss_fn(output[idx_ce], target[idx_ce])
        else:
            Z_batch = torch.ones(len(output)).long().to(self.device)
            loss_ce = self.criterion_pred_loss_fn(output, target)

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
            output = self.model(X_batch)
            loss = self.calculate_loss(output, Y_batch, None, training=False)
            pred = output.argmax(dim=1)
            acc = pred.eq(Y_batch).sum()

            loss_val += loss.item()
            acc_val += acc.item()

        loss_val /= len(val_loader)
        acc_val /= len(val_loader)

        return loss_val, acc_val

    def train(self,
              train_loader,
              save_path,
              val_loader=None,
              num_epochs=10):
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

        best_loss = None
        best_acc = None

        lr_milestones = [int(num_epochs * 0.5)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=lr_milestones, gamma=0.1)

        for epoch in tqdm(range(num_epochs)):

            self.train_epoch(train_loader)

            scheduler.step()
            self.model.eval()

            if val_loader is not None:
                epoch_loss_val, epoch_acc_val = self.validate(val_loader)

                # Early stopping by loss
                save_checkpoint = True if best_loss is None or best_loss > epoch_loss_val else False
                best_loss = epoch_loss_val if best_loss is None or best_loss > epoch_loss_val else best_loss
                if save_checkpoint:
                    self.save_checkpoint(epoch, save_path, "loss")

                # Early stopping by accuracy
                save_checkpoint = True if best_acc is None or best_acc < epoch_acc_val else False
                best_acc = epoch_acc_val if best_acc is None or best_acc < epoch_acc_val else best_acc
                if save_checkpoint:
                    self.save_checkpoint(epoch, save_path, "acc")

        self.save_checkpoint(epoch, save_path, "final")

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

        dataset = data_loader.dataset

        Z_data = torch.zeros(len(dataset)).long().to(self.device)
        split = int(len(dataset) * split_ratio)
        Z_data[torch.randperm(len(dataset))[split:]] = 1

        train_dataset = TrainDataset(dataset.X_data, dataset.Y_data, Z_data)
        train_loader = DataLoader(
            train_dataset, batch_size=data_loader.batch_size, shuffle=True, drop_last=data_loader.drop_last)
        
        return train_loader


    def save_checkpoint(self, epoch: int, save_path: str, save_type: str='final'):
        """
        Saves a checkpoint of the model and optimizer state at the given epoch.

        Args:
            epoch (int): The current epoch number.
            save_path (str): The path where the checkpoint should be saved.
            save_type (str): The type of checkpoint to save, including "final", "loss", "acc".
        """
        save_path += save_type + '.pt'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, load_path: str, load_type: str='final'):
        """
        Loads a model checkpoint from the specified path.

        Args:
            load_path (str): The path from which to load the checkpoint.
            load_type (str): The type of checkpoint to load (default: "final"), chosen from ["final", "loss", "acc"].
        """
        if not os.path.exists(load_path + load_type + '.pt'):
            load_path += "final" + '.pt'
        else:
            load_path += load_type + '.pt'
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
