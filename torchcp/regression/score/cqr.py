# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseScore
from ..loss import QuantileLoss
from ..utils import build_regression_model

class CQR(BaseScore):
    """
    Conformalized Quantile Regression (CQR) 
    This score function allows for calculating scores and generating prediction intervals
    using quantile regression model.

    Reference:
        Paper: Conformalized Quantile Regression (Romano et al., 2019)
        Link: https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf
        Github: https://github.com/yromano/cqr
    """
    def __init__(self):
        super().__init__()
        
    def __call__(self, predicts, y_truth):
        """
        Calculates the non-conformity scores for predictions.

        Non-conformity scores are defined as the absolute deviation between the true values 
        and the predicted values. These scores are used to evaluate the quality of prediction 
        intervals in conformal prediction.

        Args:
            predicts (torch.Tensor): Tensor of predicted quantile intervals, shape (batch_size, 2).
            y_truth (torch.Tensor): Tensor of true target values, shape (batch_size,).

        Returns:
            torch.Tensor: Tensor of non-conformity scores, shape (batch_size,).
        """
        if len(predicts.shape) == 2:
            predicts = predicts.unsqueeze(1)
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return torch.maximum(predicts[..., 0] - y_truth, y_truth - predicts[..., 1])
    
    def generate_intervals(self, predicts_batch, q_hat):
        """
        Generates prediction intervals for a batch of predictions.

        Prediction intervals are constructed by adding and subtracting the calibrated
        thresholds (:attr:`q_hat`) from the predicted values.

        Args:
            predicts_batch (torch.Tensor): A batch of predictions, shape (batch_size, ...).
            q_hat (torch.Tensor): Calibrated thresholds, shape (num_thresholds,).

        Returns:
            torch.Tensor: Prediction intervals, shape (batch_size, num_thresholds, 2).
                          The last dimension contains the lower and upper bounds of the intervals.
        """
        if len(predicts_batch.shape) == 2:
            predicts_batch = predicts_batch.unsqueeze(1)
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        prediction_intervals[..., 0] = predicts_batch[..., 0] - q_hat.view(1, q_hat.shape[0], 1)
        prediction_intervals[..., 1] = predicts_batch[..., 1] + q_hat.view(1, q_hat.shape[0], 1)
        return prediction_intervals
        
    def fit(self, train_dataloader, **kwargs):
        """
        Trains the model on provided training data with :math:`[alpha/2, 1-alpha/2]` quantile regression loss.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader providing training data.
            **kwargs: Additional training parameters.
                - model (torch.nn.Module, optional): Model to be trained; defaults to the model passed to the predictor.
                - criterion (callable, optional): Loss function for training. If not provided, uses :func:`QuantileLoss`.
                - alpha (float, optional): Significance level (e.g., 0.1) for quantiles, required if :attr:`criterion` is None.
                - epochs (int, optional): Number of training epochs. Default is :math:`100`.
                - lr (float, optional): Learning rate for optimizer. Default is :math:`0.01`.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training; defaults to :func:`torch.optim.Adam`.
                - verbose (bool, optional): If True, displays training progress. Default is True.

        Raises:
            ValueError: If :attr:`criterion` is not provided and :attr:`alpha` is not specified.
        """
        device = kwargs.get('device', None)
        model = kwargs.get('model', build_regression_model("NonLinearNet")(next(iter(train_dataloader))[0].shape[1], 2, 64, 0.5).to(device))
        criterion = kwargs.get('criterion', None)

        if criterion is None:
            alpha = kwargs.get('alpha', None)
            if alpha is None:
                raise ValueError("When 'criterion' is not provided, 'alpha' must be specified.")
            quantiles = [alpha / 2, 1 - alpha / 2]
            criterion = QuantileLoss(quantiles)

        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._train(model, epochs, train_dataloader, criterion, optimizer, verbose)
        return model