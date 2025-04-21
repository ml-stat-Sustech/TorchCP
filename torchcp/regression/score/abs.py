# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim as optim

from torchcp.regression.score.base import BaseScore
from torchcp.regression.utils import build_regression_model


class ABS(BaseScore):
    """
    Absolute value of the difference between prediction and true value.
    
    This score function allows for calculating scores and generating prediction intervals
    using a single-point regression model.
    
    Reference:
        Paper: Distribution-Free Predictive Inference For Regression (Lei et al., 2017)
        Link: https://arxiv.org/abs/1604.04173
        Github: https://github.com/ryantibs/conformal
    """

    def __init__(self):
        super().__init__()

    def __call__(self, predicts, y_truth):
        """
        Calculates the score used for conformal prediction, which measures the deviation 
        of the true values from the predicted intervals.

        Args:
            predicts (torch.Tensor): Tensor of predicted quantile intervals, shape (batch_size, 2).
            y_truth (torch.Tensor): Tensor of true target values, shape (batch_size,).

        Returns:
            torch.Tensor: Tensor of non-conformity scores indicating the deviation of predictions from true values.
        """
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return torch.abs(predicts - y_truth)

    def generate_intervals(self, predicts_batch, q_hat):
        """
        Generate prediction intervals by adjusting predictions with the calibrated :attr:`q_hat` threshold.
        
        Args:
            predicts_batch (torch.Tensor): A batch of predictions with shape (batch_size, ...).
            q_hat (torch.Tensor): A tensor containing the calibrated thresholds with shape (num_thresholds,).
            
        Returns:
            torch.Tensor: A tensor containing the prediction intervals with shape (batch_size, num_thresholds, 2).
                          The last dimension represents the lower and upper bounds of the intervals.
        """
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        prediction_intervals[..., 0] = predicts_batch - q_hat.view(1, q_hat.shape[0])
        prediction_intervals[..., 1] = predicts_batch + q_hat.view(1, q_hat.shape[0])
        return prediction_intervals

    def train(self, train_dataloader, **kwargs):
        """
        Trains the model using the provided training data.
        
        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            **kwargs: Additional keyword arguments for training configuration.
                - model (nn.Module, optional): The model to be trained. Defaults to the model passed to the predictor.
                - epochs (int, optional): Number of training epochs. Defaults to :math:`100`.
                - criterion (nn.Module, optional): Loss function. Defaults to :func:`torch.nn.MSELoss()`.
                - lr (float, optional): Learning rate for the optimizer. Defaults to :math:`0.01`.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training. Defaults to :func:`torch.optim.Adam`.
                - verbose (bool, optional): If True, prints training progress. Defaults to True.
        """
        device = kwargs.get('device', None)
        model = kwargs.get('model',
                           build_regression_model("NonLinearNet")(next(iter(train_dataloader))[0].shape[1], 1, 64,
                                                                  0.5).to(device))
        epochs = kwargs.get('epochs', 100)
        criterion = kwargs.get('criterion', nn.MSELoss())
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._basetrain(model, epochs, train_dataloader, criterion, optimizer, verbose)
        return model
