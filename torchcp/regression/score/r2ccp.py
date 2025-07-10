# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.optim as optim

from torchcp.regression.loss import R2ccpLoss
from torchcp.regression.score.base import BaseScore
from torchcp.regression.utils import build_regression_model


class R2CCP(BaseScore):
    """
    Regression-to-Classification Conformal Prediction.
    
    This method converting regression to a classification problem and 
    then use CP for classification to obtain CP sets for regression.

    Args:
        midpoints (torch.Tensor): the midpoints of the equidistant bins.
        
    Reference:
        Paper: Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2021)
        Link: https://neurips.cc/virtual/2023/80610
        Github: https://github.com/EtashGuha/R2CCP
    """

    def __init__(self, midpoints, device=None):
        super().__init__()
        if device is not None:
            self._device = torch.device(device)
        else:
            self._device = midpoints.device
        self.midpoints = midpoints.to(self._device)

    def train(self, train_dataloader, **kwargs):
        """
        Trains regression-to-classification model with the R2ccpLoss.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            kwargs: Additional parameters for training.
                - model (torch.nn.Module, optional): Model to be trained; defaults to the model passed to the predictor.
                - epochs (int, optional): Number of training epochs. Default is :math:`100`.
                - p (float, optional): Probability parameter for the loss function. Default is :math:`0.5`.
                - tau (float, optional): Threshold parameter for the loss function. Default is :math:`0.2`.
                - criterion (callable, optional): Loss function; defaults to :class:`R2ccpLoss`.
                - lr (float, optional): Learning rate. Default is :math:`1e-4`.
                - weight_decay (float, optional): Weight decay for the optimizer. Default is :math:`1e-4`.
                - optimizer (torch.optim.Optimizer, optional): Optimizer; defaults to :func:`torch.optim.AdamW`.
                - verbose (bool, optional): If True, displays training progress. Default is True.
                
        """
        device = kwargs.get('device', self._device)
        self._device = device
        model = kwargs.get('model', build_regression_model("NonLinearNet")(next(iter(train_dataloader))[0].shape[1],
                                                                           len(self.midpoints), 1000, 0).to(
            self._device))
        epochs = kwargs.get('epochs', 100)
        p = kwargs.get('p', 0.5)
        tau = kwargs.get('tau', 0.2)
        criterion = kwargs.get('criterion', R2ccpLoss(p, tau, self.midpoints))
        lr = kwargs.get('lr', 1e-4)
        weight_decay = kwargs.get('weight_decay', 1e-4)
        optimizer = kwargs.get('optimizer', optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay))
        verbose = kwargs.get('verbose', True)

        self._basetrain(model, epochs, train_dataloader, criterion, optimizer, verbose)
        return model

    def __call__(self, predicts, y_truth):
        interval = self.__find_interval(self.midpoints, y_truth)
        scores = self.__calculate_linear_interpolation(interval, predicts, y_truth, self.midpoints)
        return - scores  # since r2ccp calculates conformity score instead of nonconformity score

    def generate_intervals(self, predicts_batch, q_hat):
        q_hat = - q_hat  # since r2ccp calculates conformity score instead of nonconformity score
        K = predicts_batch.shape[1]
        N = predicts_batch.shape[0]
        midpoints_expanded = self.midpoints.unsqueeze(0).expand(N, K)
        left_points = midpoints_expanded[:, :-1]
        right_points = midpoints_expanded[:, 1:]
        left_predicts = predicts_batch[:, :-1]
        right_predicts = predicts_batch[:, 1:]
        q_hat_expanded = torch.ones((predicts_batch.shape[0], predicts_batch.shape[1] - 1),
                                    device=self._device) * q_hat

        mask1 = (left_predicts >= q_hat_expanded) & (right_predicts >= q_hat_expanded)
        mask2 = (left_predicts <= q_hat_expanded) & (right_predicts <= q_hat_expanded)
        mask3 = (left_predicts < q_hat_expanded) & (right_predicts > q_hat_expanded)
        mask4 = (left_predicts > q_hat_expanded) & (right_predicts < q_hat_expanded)

        prediction_intervals = torch.zeros((N, 2 * (K - 1)), device=self._device)
        prediction_intervals[:, 0::2][mask1] = left_points[mask1]
        prediction_intervals[:, 1::2][mask1] = right_points[mask1]
        prediction_intervals[:, 0::2][mask2] = 0
        prediction_intervals[:, 1::2][mask2] = 0

        delta_midpoints = right_points - left_points
        prediction_intervals[:, 0::2][mask3] = right_points[mask3] - delta_midpoints[mask3] * \
                                               (right_predicts[mask3] - q_hat_expanded[mask3]) / (
                                                       right_predicts[mask3] - left_predicts[mask3])
        prediction_intervals[:, 1::2][mask3] = right_points[mask3]

        prediction_intervals[:, 0::2][mask4] = left_points[mask4]
        prediction_intervals[:, 1::2][mask4] = left_points[mask4] - delta_midpoints[mask4] * \
                                               (left_predicts[mask4] - q_hat_expanded[mask4]) / (
                                                       right_predicts[mask4] - left_predicts[mask4])

        return prediction_intervals

    def __find_interval(self, midpoints, y_truth):
        """
        Finds the interval index for each ground truth value based on midpoints, 
        i.e., midpoints[interval[i]] <= y_truth[i] < midpoints[interval[i+1]]

        Args:
            midpoints (torch.Tensor): Midpoints of the equidistant bins.
            y_truth (torch.Tensor): Ground truth values, shape (batch_size,).

        Returns:
            torch.Tensor: Interval indices for each ground truth value, shape (batch_size,).
        """
        interval = torch.zeros_like(y_truth, dtype=torch.long).to(self._device)

        for i in range(len(midpoints) + 1):
            if i == 0:
                mask = y_truth < midpoints[i]
                interval[mask] = -1
            elif i < len(midpoints):
                mask = (y_truth >= midpoints[i - 1]) & (y_truth < midpoints[i])
                interval[mask] = i - 1
            else:
                mask = y_truth >= midpoints[-1]
                interval[mask] = len(midpoints)
        return interval

    def __calculate_linear_interpolation(self, interval, predicts, y_truth, midpoints):
        """
        Calculates scores through linear interpolation within intervals.

        Args:
            interval (torch.Tensor): Interval indices for each data point, shape (batch_size,).
            predicts (torch.Tensor): Predicted bin probabilities, shape (batch_size, num_bins).
            y_truth (torch.Tensor): Ground truth values, shape (batch_size,).
            midpoints (torch.Tensor): Midpoints of the equidistant bins.

        Returns:
            torch.Tensor: Scores for each data point, shape (batch_size,).
        """
        midpoints = midpoints.repeat((y_truth.shape[0], 1))
        left_points = midpoints[torch.arange(y_truth.shape[0]), (interval) % midpoints.shape[1]]
        right_points = midpoints[torch.arange(y_truth.shape[0]), (interval + 1) % midpoints.shape[1]]

        left_predicts = predicts[torch.arange(y_truth.shape[0]), (interval) % midpoints.shape[1]]
        right_predicts = predicts[torch.arange(y_truth.shape[0]), (interval + 1) % midpoints.shape[1]]
        scores = ((y_truth - left_points) * right_predicts + (right_points - y_truth) * left_predicts) / (
                right_points - left_points)
        scores = torch.where(interval == -1, predicts[:, 0], scores)
        scores = torch.where(interval == len(midpoints), predicts[:, -1], scores)

        if len(scores.shape) == 1:
            scores = scores.unsqueeze(-1)

        return scores
