# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.regression.predictors.split import SplitPredictor


class ACI(SplitPredictor):
    """
    Adaptive conformal inference (Gibbs et al., 2021)
    paper: https://arxiv.org/abs/2106.00170

    :param model: a pytorch model that can output the values of different quantiles.
    :param gamma: a step size parameter.
    """

    def __init__(self, model, gamma):
        super().__init__(model)
        self.__gamma = gamma
        self.alpha_t = None

    def calculate_threshold(self, predicts, y_truth, alpha):
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        self.scores = torch.maximum(predicts[:, 0] - y_truth, y_truth - predicts[:, 1])
        self.alpha = alpha
        if self.alpha_t is None:
            self.alpha_t = alpha

    def predict(self, x, y_t=None, pred_interval_t=None):
        """
        
        :param x: input features at the time t+1.
        :param y_t: the truth value at the time t.
        :param pred_interval_t: the prediction interval for the time t.
        """
        self._model.eval()
        x = x.to(self._device)

        if y_t is None:
            err_t = self.alpha
        else:
            if len(y_t.shape) == 0:
                err_t = 1 if (y_t >= pred_interval_t[0]) & (y_t <= pred_interval_t[1]) else 0
            else:
                steps_t = len(y_t)
                w = torch.arange(steps_t).to(self._device)
                w = torch.pow(0.95, w)
                w = w / torch.sum(w)
                err = x.new_zeros(steps_t)
                for i in range(steps_t):
                    err[i] = 1 if (y_t[i] >= pred_interval_t[i][0]) & (y_t[i] <= pred_interval_t[i][1]) else 0
                err_t = torch.sum(w * err)
        self.alpha_t = self.alpha_t + self.__gamma * (self.alpha - err_t)
        predicts_batch = self._model(x.to(self._device)).float()
        quantile = (1 - self.alpha_t) * (1 + 1 / self.scores.shape[0])
        if quantile > 1:
            quantile = 1
        q_hat = torch.quantile(self.scores, quantile)
        prediction_intervals = x.new_zeros(2)
        prediction_intervals[0] = predicts_batch[0] - q_hat
        prediction_intervals[1] = predicts_batch[1] + q_hat
        return prediction_intervals
