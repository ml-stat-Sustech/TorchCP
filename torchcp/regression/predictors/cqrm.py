# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .cqr import CQR
from ..loss import QuantileLoss


class CQRM(CQR):
    """
    Method: CQR-M
    Paper: A comparison of some conformal quantile regression methods (Matteo Sesia and Emmanuel J. Candes, 2019)
    Link: https://onlinelibrary.wiley.com/doi/epdf/10.1002/sta4.261
    Github: https://github.com/msesia/cqr-comparison

    :param model: a pytorch model that can output alpha/2, 1/2 and 1-alpha/2 quantile regression.
    """

    def fit(self, train_dataloader, **kwargs):
        criterion = kwargs.pop('criterion', None)
        if criterion is None:
            alpha = kwargs.pop('alpha', None)
            if alpha is None:
                raise ValueError("When 'criterion' is not provided, 'alpha' must be specified.")
            quantiles = [alpha / 2, 1 / 2, 1 - alpha / 2]
            criterion = QuantileLoss(quantiles)
        super().fit(train_dataloader, criterion=criterion, **kwargs)

    def calculate_score(self, predicts, y_truth):
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
