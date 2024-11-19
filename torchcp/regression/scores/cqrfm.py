# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .cqrm import CQRM


class CQRFM(CQRM):
    """
    Conformal Quantile Regression Fraction Median

    Reference:
        Paper: Adaptive, Distribution-Free Prediction Intervals for Deep Networks (Kivaranovic et al., 2019)
        Link: https://proceedings.mlr.press/v108/kivaranovic20a.html
    """

    def __call__(self, predicts, y_truth):
        if len(predicts.shape) == 2:
            predicts = predicts.unsqueeze(1)
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return torch.maximum((predicts[..., 1] - y_truth) / (predicts[..., 1] - predicts[..., 0]),
                             (y_truth - predicts[..., 1]) / (predicts[..., 2] - predicts[..., 1]))

    def generate_intervals(self, predicts_batch, q_hat):
        if len(predicts_batch.shape) == 2:
            predicts_batch = predicts_batch.unsqueeze(1)
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))

        prediction_intervals[..., 0] = predicts_batch[..., 1] - q_hat.view(1, q_hat.shape[0], 1) * \
                                       (predicts_batch[..., 1] - predicts_batch[..., 0])
        prediction_intervals[..., 1] = predicts_batch[..., 1] + q_hat.view(1, q_hat.shape[0], 1) * \
                                       (predicts_batch[..., 2] - predicts_batch[..., 1])
        return prediction_intervals
