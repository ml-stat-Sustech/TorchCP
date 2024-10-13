# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.utils.common import calculate_conformal_value
from .base import BaseGraphPredictor


class GraphSplitPredictor(BaseGraphPredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.

    :param score_function: graph non-conformity score function.
    :param model: a pytorch model.
    """

    def __init__(self, score_function, model=None, graph_data=None):
        super().__init__(score_function, model, graph_data)

    def calculate_threshold(self, logits, cal_idx, label_mask, alpha):
        scores = self.score_function(logits).to(self._device)
        label_mask = label_mask.to(self._device)

        cal_scores = scores[cal_idx][label_mask[cal_idx]]
        self.q_hat = self._calculate_conformal_value(cal_scores, alpha)

    def _calculate_conformal_value(self, scores, alpha, marginal_q_hat=torch.inf):
        return calculate_conformal_value(scores, alpha, marginal_q_hat)

    def predict_with_logits(self, logits, eval_idx, q_hat=None):
        """
        The input of score function is softmax probability.
        if q_hat is not given by the function 'self.calibrate', the construction progress of prediction set is a naive method.

        :param logits: model output before softmax.
        :param eval_idx: indices of calibration set.
        :param q_hat: the conformal threshold.

        :return: prediction sets
        """
        scores = self.score_function(logits).to(self._device)

        eval_scores = scores[eval_idx]
        if q_hat is None:
            q_hat = self.q_hat

        S = self._generate_prediction_set(eval_scores, q_hat)

        return S

    def predict_with_scores(self, scores, q_hat=None):
        if q_hat is None:
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)
        return S
    
    def calibrate(self, cal_idx, alpha):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._graph_data.x, self._graph_data.edge_index)
        self.calculate_threshold(logits, cal_idx, self._label_mask, alpha)

    def evaluate(self, eval_idx):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._graph_data.x, self._graph_data.edge_index)
        prediction_sets = self.predict_with_logits(logits, eval_idx)

        res_dict = {"Coverage_rate": self._metric('coverage_rate')(prediction_sets, self._graph_data.y[eval_idx]),
                    "Average_size": self._metric('average_size')(prediction_sets, self._graph_data.y[eval_idx])}
        return res_dict
