# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch

from torchcp.regression.predictors.split import SplitPredictor


class CQR(SplitPredictor):
    """
    Conformalized Quantile Regression (Romano et al., 2019)
    paper: https://arxiv.org/abs/1905.03222

    :param model: a pytorch model that can output alpha/2 and 1-alpha/2 quantile regression.
    """

    def __init__(self, model):
        super().__init__(model)
        
    def calculate_score(self, predicts, y_truth):
        return torch.maximum(predicts[:, 0] - y_truth, y_truth - predicts[:, 1])

    def predict(self, x_batch):
        self._model.eval()
        predicts_batch = self._model(x_batch.to(self._device)).float()
        if len(x_batch.shape) == 1:
            x_batch = x_batch.unsqueeze(0)

        predicts_batch = self._model(x_batch.to(self._device)).float()
        prediction_intervals = x_batch.new_zeros((x_batch.shape[0], 2))
        prediction_intervals[:, 0] = predicts_batch[:, 0] - self.q_hat
        prediction_intervals[:, 1] = predicts_batch[:, 1] + self.q_hat
        return prediction_intervals
