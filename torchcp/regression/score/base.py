# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta, abstractmethod
from tqdm import tqdm

from torchcp.utils.common import get_device


class BaseScore(object):
    """
    Abstract base class for all score functions.
    """
    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, logits, labels):
        """Virtual method to compute scores for a data pair (x,y).

        Args:
            logits: the logits for inputs.
            labels: the labels.
        """
        raise NotImplementedError

    @abstractmethod
    def construct_interval(self, predicts_batch, q_hat):
        """Constructs the prediction interval for the given batch of predictions.

        Args:
            predicts_batch: the batch of predictions.
            q_hat: the quantile level.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, model, epochs, train_dataloader, criterion, optimizer, verbose=True):
        """Trains the given model using the provided training data loader, criterion, and optimizer.

        Args:
            model: the model to be trained.
            epochs: the number of epochs to train the model.
            train_dataloader: DataLoader for the training data.
            criterion: the loss function.
            optimizer: the optimizer for updating the model parameters.
            verbose: if True, displays a progress bar and loss information.
        """
        raise NotImplementedError

    def _basetrain(self, model, epochs, train_dataloader, criterion, optimizer, verbose=True):
        """
        Trains the given model using the provided training data loader, criterion, and optimizer.
        
        Args:
            model (torch.nn.Module): The model to be trained.
            epochs (int): The number of epochs to train the model.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
            criterion (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer for updating the model parameters.
            verbose (bool, optional): If True, displays a progress bar and loss information. Defaults to True.
        """

        model.train()
        device = get_device(model)
        if verbose:
            with tqdm(total=epochs, desc="Epoch") as _tqdm:
                for epoch in range(epochs):
                    running_loss = 0.0
                    for index, (tmp_x, tmp_y) in enumerate(train_dataloader):
                        outputs = model(tmp_x.to(device))
                        loss = criterion(outputs, tmp_y.reshape(-1, 1).to(device))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        running_loss = (running_loss * max(0, index) + loss.data.cpu().numpy()) / (index + 1)
                        _tqdm.set_postfix({"loss": f"{running_loss:.6f}"})
                    _tqdm.update(1)
        else:
            for tmp_x, tmp_y in train_dataloader:
                outputs = model(tmp_x.to(device))
                loss = criterion(outputs, tmp_y.reshape(-1, 1).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("Training complete.")
        model.eval()
