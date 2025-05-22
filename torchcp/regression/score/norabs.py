# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim as optim

from torchcp.regression.score.abs import ABS
from torchcp.regression.utils import build_regression_model


class NorABS(ABS):
    """
    Normalized Absolute Score (NorABS) used for conformal regression.

    This score function computes the absolute difference between the predicted mean and 
    the ground truth value, normalized by the predicted standard deviation. It is designed 
    for use with probabilistic regression models that predict both the mean and variance.

    Reference:
        Book: Algorithmic Learning in a Random World (Vovk et al., 2005)
        Link: https://link.springer.com/book/10.1007/b106715
    """

    def __init__(self):
        super().__init__()

    def __call__(self, predicts, y_truth):
        """
        Computes the normalized non-conformity score for conformal prediction.

        Args:
            predicts (torch.Tensor): Tensor containing predicted mean and standard deviation,
                                     shape (batch_size, 2), where the first column is mean (μ)
                                     and the second column is standard deviation (σ).
            y_truth (torch.Tensor): Tensor of true target values, shape (batch_size,).

        Returns:
            torch.Tensor: Tensor of normalized absolute deviations, shape (batch_size, 1).
        """
        mu, var = predicts[..., 0], predicts[..., 1]
        scores = torch.abs(mu - y_truth) / var
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(1)
        return scores

    def generate_intervals(self, predicts_batch, q_hat):
        """
        Generates prediction intervals using predicted means and standard deviations, 
        scaled by the calibrated threshold :attr:`q_hat`.

        Args:
            predicts_batch (torch.Tensor): Tensor of predicted (mean, std), shape (batch_size, 2).
            q_hat (torch.Tensor): Calibrated threshold values, shape (num_thresholds,).

        Returns:
            torch.Tensor: Prediction intervals, shape (batch_size, num_thresholds, 2),
                          where the last dimension contains lower and upper bounds.
        """
        if len(predicts_batch.shape) == 2:
            predicts_batch = predicts_batch.unsqueeze(1)
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        prediction_intervals[..., 0] = predicts_batch[..., 0] - q_hat.view(1, q_hat.shape[0]) * predicts_batch[..., 1]
        prediction_intervals[..., 1] = predicts_batch[..., 0] + q_hat.view(1, q_hat.shape[0]) * predicts_batch[..., 1]
        return prediction_intervals

    def train(self, train_dataloader, **kwargs):
        """
        Trains the probabilistic regression model to predict both mean and variance.

        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            **kwargs: Additional keyword arguments for training configuration.
                - model (nn.Module, optional): Custom regression model. If None, defaults to
                                               GaussianRegressionModel.
                - epochs (int, optional): Number of training epochs. Defaults to 100.
                - criterion (nn.Module, optional): Loss function. Defaults to GaussianNLLLoss.
                - lr (float, optional): Learning rate. Defaults to 0.01.
                - optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam.
                - verbose (bool, optional): Whether to print training progress. Defaults to True.

        Returns:
            nn.Module: The trained regression model.
        """
        device = kwargs.get('device', None)
        model = kwargs.get('model',
                           build_regression_model("GaussianRegressionModel")(next(iter(train_dataloader))[0].shape[1], 64,
                                                                              0.5).to(device))
        epochs = kwargs.get('epochs', 100)
        criterion = kwargs.get('criterion', nn.GaussianNLLLoss())
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._basetrain(model, epochs, train_dataloader, criterion, optimizer, verbose)
        return model
