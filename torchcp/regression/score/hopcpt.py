# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.regression.score.abs import ABS
from torchcp.regression.utils import build_regression_model

class HopCPT(ABS):
    def __init__(self):
        super().__init__()
    
    def __call__(self, predicts, y_truth):
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return y_truth - predicts
    
    def generate_intervals(self, predicts_batch, q_hat):
        """
        Generate prediction intervals by adjusting predictions with the calibrated :attr:`q_hat` threshold.

        Args:
            predicts_batch (torch.Tensor): A batch of predictions with shape (batch_size, ...).
            q_hat (torch.Tensor): A tensor containing the calibrated thresholds with shape (batch_size, 2).

        Returns:
            torch.Tensor: A tensor containing the prediction intervals with shape (batch_size, num_thresholds, 2).
                        The last dimension represents the lower and upper bounds of the intervals.
        """
        prediction_intervals = predicts_batch.unsqueeze(-1) + q_hat.view(-1, 1, 2)

        return prediction_intervals
