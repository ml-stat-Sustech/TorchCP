# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .cqr import CQR


class CQRR(CQR):
    """
    A comparison of some conformal quantile regression methods (Matteo Sesia and Emmanuel J. Candes, 2019)
    paper: https://onlinelibrary.wiley.com/doi/epdf/10.1002/sta4.261

    :param model: a pytorch model that can output alpha/2 and 1-alpha/2 quantile regression.
    """

    def calculate_score(self, predicts, y_truth):
        if len(predicts.shape) == 2:
            predicts = predicts.unsqueeze(1)
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        eps = 1e-6
        scaling_factor = predicts[..., 1] - predicts[..., 0] + eps
        return torch.maximum((predicts[..., 0] - y_truth) / scaling_factor,
                             (y_truth - predicts[..., 1]) / scaling_factor)

    def generate_intervals(self, predicts_batch, q_hat):
        if len(predicts_batch.shape) == 2:
            predicts_batch = predicts_batch.unsqueeze(1)
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))

        eps = 1e-6
        scaling_factor = prediction_intervals[..., 1] - prediction_intervals[..., 0] + eps
        prediction_intervals[..., 0] = predicts_batch[..., 0] - q_hat.view(1, q_hat.shape[0], 1) * scaling_factor
        prediction_intervals[..., 1] = predicts_batch[..., 1] + q_hat.view(1, q_hat.shape[0], 1) * scaling_factor
        return prediction_intervals