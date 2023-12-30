# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import torch.nn.functional as F

from torchcp.regression.predictors.split import SplitPredictor
from torchcp.regression.utils.metrics import Metrics
from torchcp.utils.common import get_device, calculate_conformal_value

class R2CCP(SplitPredictor):
    """
    Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2021)
    paper: https://neurips.cc/virtual/2023/80610

    :param model: a pytorch model that can output probabilities for different bins.
    :param K: number of bins.
    """
    
    def __init__(self, model, K, midpoints):
        super().__init__(model)
        self._model = model
        self._device = get_device(model)
        self._metric = Metrics()
        self.q_hat = None
        self.alpha = None
        self.K = K
        self.midpoints = midpoints

    def calculate_threshold(self, predicts, y_truth, alpha):
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        
        # linear interpolation of softmax probabilities
        interval = self.__find_interval(self.midpoints, y_truth)
        scores = self.__calculate_linear_interpolation(interval, predicts, y_truth, self.midpoints)
        # scores = F.interpolate(y.unsqueeze(0), new_x.view(1, -1), mode='linear', align_corners=False)

        self.q_hat = calculate_conformal_value(scores, alpha)

    def predict(self, x_batch):
        self._model.eval()
        midpoints = self.midpoints

        predicts_batch = self._model(x_batch.to(self._device)).float()
        prediction_intervals = x_batch.new_zeros((x_batch.shape[0], 2*(self.K-1)))
        with torch.no_grad():
            for i in range(len(x_batch)):
                for k in range(self.K-1):
                    if predicts_batch[i, k] >= self.q_hat and predicts_batch[i, k + 1] >= self.q_hat:
                        prediction_intervals[i, 2 * k] = midpoints[k]
                        prediction_intervals[i, 2 * k + 1] = midpoints[k + 1]
                    elif predicts_batch[i, k] <= self.q_hat and predicts_batch[i, k + 1] <= self.q_hat:
                        prediction_intervals[i, 2 * k] = midpoints[k + 1]
                        prediction_intervals[i, 2 * k + 1] = midpoints[k + 1]
                    elif predicts_batch[i, k] <= self.q_hat and predicts_batch[i, k + 1] >= self.q_hat:
                        prediction_intervals[i, 2 * k] = midpoints[k] + (midpoints[k + 1] - midpoints[k]) * \
                                (self.q_hat - predicts_batch[i, k]) / (predicts_batch[i, k + 1] - predicts_batch[i, k])
                        prediction_intervals[i, 2 * k + 1] = midpoints[k + 1]
                    else:
                        prediction_intervals[i, 2 * k] = midpoints[k]
                        prediction_intervals[i, 2 * k + 1] = midpoints[k] - (midpoints[k + 1] - midpoints[k]) * \
                                (predicts_batch[i, k] - self.q_hat) / (predicts_batch[i, k + 1] - predicts_batch[i, k])
        return prediction_intervals
    
    def __find_interval(self, midpoints, y_truth):
        interval = torch.zeros_like(y_truth, dtype=torch.long)

        for i in range(len(midpoints)):
            if i == 0:
                mask = y_truth < midpoints[i]
                interval[mask] = 0
            elif i < len(midpoints):
                mask = (y_truth >= midpoints[i-1]) & (y_truth < midpoints[i])
                interval[mask] = i
            else:
                mask = y_truth >= midpoints[-1]
                interval[mask] = len(midpoints)
        return interval

    
    def __calculate_linear_interpolation(self, interval, predicts, y_truth, midpoints):
        scores = torch.zeros_like(y_truth, dtype=torch.float32)
        
        for i in range(len(y_truth)):
            k = interval[i]
            score = (predicts[i,k+1] - y_truth[i]) * predicts[i,k] / (midpoints[k+1] - midpoints[k]) \
                + (y_truth[i] - predicts[i,k]) * predicts[i,k+1] / (midpoints[k+1] - midpoints[k])
            scores[i] = score
            
        return scores