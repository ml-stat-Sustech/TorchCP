# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.regression.predictors.split import SplitPredictor

class R2CCP(SplitPredictor):
    """
    Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2021)
    paper: https://neurips.cc/virtual/2023/80610

    :param model: a pytorch model that can output probabilities for different bins.
    :param midpoints: the midpoint of each bin.
    """
    
    def __init__(self, model, midpoints):
        super().__init__(model)
        self.midpoints = midpoints

    def calculate_score(self, predicts, y_truth):
        interval = self.__find_interval(self.midpoints, y_truth)
        scores = self.__calculate_linear_interpolation(interval, predicts, y_truth, self.midpoints)
        return scores
    
    def calculate_threshold(self, predicts, y_truth, alpha):
        scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(scores, 1-alpha)

    def predict(self, x_batch):
        self._model.eval()
        midpoints = self.midpoints
        K = len(midpoints)

        predicts_batch = self._model(x_batch.to(self._device)).float()
        prediction_intervals = x_batch.new_zeros((x_batch.shape[0], 2*(K-1))).to(self._device)
        for i in range(len(x_batch)):
            for k in range(K-1):
                if predicts_batch[i, k] >= self.q_hat and predicts_batch[i, k + 1] >= self.q_hat:
                    prediction_intervals[i, 2 * k] = midpoints[k]
                    prediction_intervals[i, 2 * k + 1] = midpoints[k + 1]
                elif predicts_batch[i, k] <= self.q_hat and predicts_batch[i, k + 1] <= self.q_hat:
                    prediction_intervals[i, 2 * k] = 0
                    prediction_intervals[i, 2 * k + 1] = 0
                elif predicts_batch[i, k] < self.q_hat and predicts_batch[i, k + 1] > self.q_hat:
                    prediction_intervals[i, 2 * k] = midpoints[k+1] - (midpoints[k + 1] - midpoints[k]) * \
                            (predicts_batch[i, k + 1] - self.q_hat) / (predicts_batch[i, k + 1] - predicts_batch[i, k])
                    prediction_intervals[i, 2 * k + 1] = midpoints[k + 1]
                elif predicts_batch[i, k] > self.q_hat and predicts_batch[i, k + 1] < self.q_hat:
                    prediction_intervals[i, 2 * k] = midpoints[k]
                    prediction_intervals[i, 2 * k + 1] = midpoints[k] - (midpoints[k + 1] - midpoints[k]) * \
                            (predicts_batch[i, k] - self.q_hat) / (predicts_batch[i, k + 1] - predicts_batch[i, k])
        return prediction_intervals
    
    def __find_interval(self, midpoints, y_truth):
        '''
        midpoints[interval[i]] <= y_truth[i] < midpoints[interval[i+1]]
        '''
        interval = torch.zeros_like(y_truth, dtype=torch.long).to(self._device)

        for i in range(len(midpoints)+1):
            if i == 0:
                mask = y_truth < midpoints[i]
                interval[mask] = 0
            elif i < len(midpoints):
                mask = (y_truth >= midpoints[i-1]) & (y_truth < midpoints[i])
                interval[mask] = i-1
            else:
                mask = y_truth >= midpoints[-1]
                interval[mask] = len(midpoints)
        return interval

    
    def __calculate_linear_interpolation(self, interval, predicts, y_truth, midpoints):
        scores = torch.zeros_like(y_truth, dtype=torch.float32).to(self._device)
        for i in range(len(y_truth)):
            k = interval[i]
            if k == 0:
                scores[i] = predicts[i,k]
            elif k == len(midpoints):
                scores[i] = predicts[i,k-1]
            else:
                scores[i] = ((midpoints[k+1] - y_truth[i]) * predicts[i,k] + (y_truth[i] - midpoints[k]) * predicts[i,k+1]) \
                        / (midpoints[k+1] - midpoints[k])
                        
        return scores