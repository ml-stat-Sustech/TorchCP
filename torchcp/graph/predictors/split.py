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

    def __init__(self, graph_data, score_function, model=None):
        super().__init__(graph_data, score_function, model)

    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_idx, alpha):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._graph_data.x, self._graph_data.edge_index)
        self.calculate_threshold(logits, cal_idx, self._label_mask, alpha)

    def calculate_threshold(self, logits, cal_idx, label_mask, alpha):
        self._device = logits.device
        
        scores = self.score_function(logits).to(self._device)
        label_mask = label_mask.to(self._device)

        cal_scores = scores[cal_idx][label_mask[cal_idx]]
        self.q_hat = self._calculate_conformal_value(cal_scores, alpha)

    def _calculate_conformal_value(self, scores, alpha, marginal_q_hat=torch.inf):
        return calculate_conformal_value(scores, alpha, marginal_q_hat)
    
    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x_batch, self._graph_data.edge_index)
        sets = self.predict_with_logits(logits)
        return sets

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
            assert self.q_hat is not None, "Ensure self.q_hat is not None. Please perform calibration first."
            q_hat = self.q_hat

        S = self._generate_prediction_set(eval_scores, q_hat)

        return S

    #############################
    # The evaluation process
    ############################

    def evaluate(self, eval_idx):
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._graph_data.x, self._graph_data.edge_index)
        prediction_sets = self.predict_with_logits(logits, eval_idx)

        res_dict = {"Coverage_rate": self._metric('coverage_rate')(prediction_sets, self._graph_data.y[eval_idx]),
                    "Average_size": self._metric('average_size')(prediction_sets, self._graph_data.y[eval_idx]),
                    "Singleton_Hit_Ratio": self._metric('singleton_hit_ratio')(prediction_sets, self._graph_data.y[eval_idx])}
        return res_dict
