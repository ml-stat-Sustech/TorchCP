# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import torch
import warnings

from torchcp.utils.common import calculate_conformal_value
from .base import BaseGraphPredictor


class SplitPredictor(BaseGraphPredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.

    :param base_score_function: basic non-conformity score function.
    :param graph_score_function: graph non-conformity score function.
    """

    def __init__(self, base_score_function, graph_score_function, model=None):
        super().__init__(base_score_function, graph_score_function, model)

    def calculate_threshold(self, embeddings, cal_idx, label_mask, alpha, n_vertices, edge_index, edge_weight=None):
        base_scores = self.base_score_function(embeddings).to(self._device)
        graph_scores = self.graph_score_function(
            base_scores, n_vertices, edge_index, edge_weight)
        label_mask = label_mask.to(self._device)

        cal_scores = graph_scores[cal_idx][label_mask[cal_idx]]
        self.q_hat = self._calculate_conformal_value(cal_scores, alpha)

    def _calculate_conformal_value(self, scores, alpha, marginal_q_hat=torch.inf):
        return calculate_conformal_value(scores, alpha, marginal_q_hat)

    def predict_with_logits(self, embeddings, eval_idx, n_vertices, edge_index, edge_weight=None, q_hat=None):
        """
        The input of score function is softmax probability.
        if q_hat is not given by the function 'self.calibrate', the construction progress of prediction set is a naive method.

        :param logits: model output before softmax.
        :param q_hat: the conformal threshold.

        :return: prediction sets
        """

        base_scores = self.base_score_function(embeddings).to(self._device)
        graph_scores = self.graph_score_function(
            base_scores, n_vertices, edge_index, edge_weight)

        eval_scores = graph_scores[eval_idx]
        if q_hat is None:
            q_hat = self.q_hat

        S = self._generate_prediction_set(eval_scores, q_hat)

        return S

    def predict_with_scores(self, scores, q_hat=None):
        if q_hat is None:
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)
        return S
