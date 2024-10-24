# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .base import BaseScore


class SNAPS(BaseScore):
    """
    Method: Similarity-Navigated Adaptive Prediction Sets
    Paper: Similarity-Navigated Conformal Prediction for Graph Neural Networks (Song et al., 2024)
    Link: https://arxiv.org/pdf/2405.14303
    Github:

    :param lambda_val: the parameter of neighborhood-based scores.
    :param mu_val: the parameter of similarity-based scores.
    """

    def __init__(self, lambda_val, mu_val, base_score_function, graph_data, knn_edge=None, knn_weight=None):
        super(SNAPS, self).__init__(base_score_function, graph_data)
        if lambda_val < 0 and lambda_val > 1:
            raise ValueError(
                "The parameter 'lambda_val' must be a value between 0 and 1.")
        if mu_val < 0 and mu_val > 1:
            raise ValueError(
                "The parameter 'mu_val' must be a value between 0 and 1.")
        if lambda_val + mu_val > 1:
            raise ValueError(
                "The summation of 'lambda_val' and 'mu_val' must not be greater than 1.")

        self._lambda_val = lambda_val
        self._mu_val = mu_val

        if knn_edge is not None:
            if knn_weight is None:
                knn_weight = torch.ones(
                    knn_edge.shape[1]).to(self._device)
            self._adj_knn = torch.sparse_coo_tensor(
                knn_edge,
                knn_weight,
                (self._n_vertices, self._n_vertices))
            self._knn_degs = torch.matmul(self._adj_knn, torch.ones(
                (self._adj_knn.shape[0])).to(self._device))
        else:
            self._adj_knn = None

    def __call__(self, logits):
        base_scores = self._base_score_function(logits)

        similarity_scores = 0.
        if self._adj_knn is not None:
            similarity_scores = torch.linalg.matmul(
                self._adj_knn, base_scores) * (1 / (self._knn_degs + 1e-10))[:, None]

        neigh_scores = torch.linalg.matmul(
            self._adj, base_scores) * (1 / (self._degs + 1e-10))[:, None]

        scores = (1 - self._lambda_val - self._mu_val) * base_scores + \
            self._lambda_val * similarity_scores + \
            self._mu_val * neigh_scores

        return scores
