# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.optim as optim

from .cqr import CQR
from ..loss import QuantileLoss
from ..utils import build_regression_model


class CQRM(CQR):
    """
    CQR-M

    Reference:
        Paper: A comparison of some conformal quantile regression methods (Matteo Sesia and Emmanuel J. Candes, 2019)
        Link: https://onlinelibrary.wiley.com/doi/epdf/10.1002/sta4.261
        Github: https://github.com/msesia/cqr-comparison
    """

    def train(self, train_dataloader, **kwargs):
        """
        Trains the model on provided training data with :math:`[alpha/2, 1/2, 1-alpha/2]` quantile regression loss.

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
            
        .. note::
            This function is optional but recommended, because the training process for each preditor's model is different. 
            We provide a default training method, and users can change the hyperparameters :attr:`kwargs` to modify the training process.
            If the train function is not used, users should pass the trained model to the predictor at the beginning.
        """
        device = kwargs.pop('device', None)
        model = kwargs.pop('model', build_regression_model("NonLinearNet")(next(iter(train_dataloader))[0].shape[1], 3, 64, 0.5).to(device))
        criterion = kwargs.pop('criterion', None)
        if criterion is None:
            alpha = kwargs.pop('alpha', None)
            if alpha is None:
                raise ValueError("When 'criterion' is not provided, 'alpha' must be specified.")
            quantiles = [alpha / 2, 1 / 2, 1 - alpha / 2]
            criterion = QuantileLoss(quantiles)
        
        epochs = kwargs.get('epochs', 100)
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._basetrain(model, epochs, train_dataloader, criterion, optimizer, verbose)
        return model

    def __call__(self, predicts, y_truth):
        if len(predicts.shape) == 2:
            predicts = predicts.unsqueeze(1)
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        eps = 1e-6
        scaling_factor_lower = predicts[..., 1] - predicts[..., 0] + eps
        scaling_factor_upper = predicts[..., 2] - predicts[..., 1] + eps
        return torch.maximum((predicts[..., 0] - y_truth) / scaling_factor_lower,
                             (y_truth - predicts[..., 2]) / scaling_factor_upper)

    def generate_intervals(self, predicts_batch, q_hat):
        if len(predicts_batch.shape) == 2:
            predicts_batch = predicts_batch.unsqueeze(1)
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        eps = 1e-6
        scaling_factor_lower = predicts_batch[..., 1] - predicts_batch[..., 0] + eps
        scaling_factor_upper = predicts_batch[..., 2] - predicts_batch[..., 1] + eps
        prediction_intervals[..., 0] = predicts_batch[..., 0] - \
                                       q_hat.view(1, q_hat.shape[0], 1) * scaling_factor_lower
        prediction_intervals[..., 1] = predicts_batch[..., 2] + \
                                       q_hat.view(1, q_hat.shape[0], 1) * scaling_factor_upper
        return prediction_intervals
