# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torchcp.regression.score.base import BaseScore


class Sign(BaseScore):
    """
    Value of the difference between prediction and true value.
    
    This score function allows for calculating scores.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, predicts, y_truth):
        """
        Calculates the score used for conformal prediction, which measures the difference 
        of the true values from the predicted intervals.

        Args:
            predicts (torch.Tensor): Tensor of predicted quantile intervals, shape (batch_size, ).
            y_truth (torch.Tensor): Tensor of true target values, shape (batch_size,).

        Returns:
            torch.Tensor: Tensor of non-conformity scores indicating the difference of predictions from true values.
        """
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return y_truth - predicts
