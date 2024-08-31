# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .split import SplitPredictor


class R2CCP(SplitPredictor):
    """
    Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2021)
    paper: https://neurips.cc/virtual/2023/80610

    :param model: a pytorch model that can output probabilities for different bins.
    :param midpoints: the midpoints of the equidistant bins.
    """

    def __init__(self, model, midpoints):
        super().__init__(model)
        self.midpoints = midpoints.to(self._device)

    def calculate_score(self, predicts, y_truth):
        interval = self.__find_interval(self.midpoints, y_truth)
        scores = self.__calculate_linear_interpolation(interval, predicts, y_truth, self.midpoints)
        return scores

    def calculate_threshold(self, predicts, y_truth, alpha):
        scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(scores, 1 - alpha)

    def predict(self, x_batch):
        self._model.eval()        
        predicts_batch = self._model(x_batch.to(self._device)).float()
        K = predicts_batch.shape[1]
        N = predicts_batch.shape[0]
        midpoints_expanded = self.midpoints.unsqueeze(0).expand(N, K)
        left_points = midpoints_expanded[:, :-1]
        right_points = midpoints_expanded[:, 1:]
        left_predicts = predicts_batch[:, :-1]
        right_predicts = predicts_batch[:, 1:]
        q_hat_expanded = torch.ones((predicts_batch.shape[0], predicts_batch.shape[1]-1), device=self._device)* self.q_hat

        mask1 = (left_predicts >= q_hat_expanded) & (right_predicts >= q_hat_expanded)
        mask2 = (left_predicts <= q_hat_expanded) & (right_predicts <= q_hat_expanded)
        mask3 = (left_predicts < q_hat_expanded) & (right_predicts > q_hat_expanded)
        mask4 = (left_predicts > q_hat_expanded) & (right_predicts < q_hat_expanded)
        
        prediction_intervals = torch.zeros((N, 2 * (K - 1)), device=self._device)
        prediction_intervals[:, 0::2][mask1] = left_points[mask1]
        prediction_intervals[:, 1::2][mask1] =right_points[mask1]
        prediction_intervals[:, 0::2][mask2] = 0
        prediction_intervals[:, 1::2][mask2] = 0

        delta_midpoints =right_points - left_points
        prediction_intervals[:, 0::2][mask3] =right_points[mask3] - delta_midpoints[mask3] * \
            (right_predicts[mask3] - q_hat_expanded[mask3]) / (right_predicts[mask3] - left_predicts[mask3])
        prediction_intervals[:, 1::2][mask3] =right_points[mask3]

        prediction_intervals[:, 0::2][mask4] = left_points[mask4]
        prediction_intervals[:, 1::2][mask4] = left_points[mask4] - delta_midpoints[mask4] * \
            (left_predicts[mask4] - q_hat_expanded[mask4]) / (right_predicts[mask4] - left_predicts[mask4])

        return prediction_intervals

    def __find_interval(self, midpoints, y_truth):
        """
        Choosing an interval for y_truth[i], i.e., midpoints[interval[i]] <= y_truth[i] < midpoints[interval[i+1]]

        :param midpoints: the midpoints of the equidistant bins
        :param y_truth: the truth values
        :return:
        """
        interval = torch.zeros_like(y_truth, dtype=torch.long).to(self._device)

        for i in range(len(midpoints) + 1):
            if i == 0:
                mask = y_truth < midpoints[i]
                interval[mask] = -1
            elif i < len(midpoints):
                mask = (y_truth >= midpoints[i - 1]) & (y_truth < midpoints[i])
                interval[mask] = i - 1
            else:
                mask = y_truth >= midpoints[-1]
                interval[mask] = len(midpoints)
        return interval

    def __calculate_linear_interpolation(self, interval, predicts, y_truth, midpoints):
        
        midpoints = midpoints.repeat((y_truth.shape[0],1))
        left_points = midpoints[torch.arange(y_truth.shape[0]), (interval)%midpoints.shape[1]]
        right_points = midpoints[torch.arange(y_truth.shape[0]), (interval+1)%midpoints.shape[1]]
        
        left_predicts = predicts[torch.arange(y_truth.shape[0]), (interval)%midpoints.shape[1]]
        right_predicts = predicts[torch.arange(y_truth.shape[0]), (interval+1)%midpoints.shape[1]]

        scores = ((y_truth-left_points)*right_predicts+(right_points - y_truth)*left_predicts)/(right_points-left_points)
        scores = torch.where(interval == -1, predicts[:,0], scores)
        scores = torch.where(interval == len(midpoints), predicts[:,-1], scores)

        return scores

