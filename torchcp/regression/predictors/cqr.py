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

    :param model: a deep learning model that can output alpha/2 and 1-alpha/2 quantile regression.
    """

    def __init__(self, model):
        super().__init__(model)

    def calculate_threshold(self, predicts, y_truth, alpha):
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        self.scores = torch.maximum(predicts[:, 0] - y_truth, y_truth - predicts[:, 1])
        quantile = math.ceil((self.scores.shape[0] + 1) * (1 - alpha)) / self.scores.shape[0]
        if quantile > 1:
            quantile = 1
        self.q_hat = torch.quantile(self.scores, quantile)

    def predict(self, x_batch):
        self._model.eval()
        predicts_batch = self._model(x_batch.to(self._device)).float()
        if len(x_batch.shape) == 2:
            predicts_batch = self._model(x_batch.to(self._device)).float()
            prediction_intervals = x_batch.new_zeros((x_batch.shape[0], 2))
            prediction_intervals[:, 0] = predicts_batch[:, 0] - self.q_hat
            prediction_intervals[:, 1] = predicts_batch[:, 1] + self.q_hat
        else:
            prediction_intervals = torch.zeros(2)
            prediction_intervals[0] = predicts_batch[0] - self.q_hat
            prediction_intervals[1] = predicts_batch[1] + self.q_hat
        return prediction_intervals
