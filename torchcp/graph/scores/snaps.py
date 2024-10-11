# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import Tensor
from torch_geometric.typing import SparseTensor

from .base import BaseScore


class SNAPS(BaseScore):
    """
    Similarity-Navigated Adaptive Prediction Sets (song et al., 2024)
    paper: https://arxiv.org/pdf/2405.14303

    :param lambda_val: the parameter of neighborhood-based scores.
    :param mu_val: the parameter of similarity-based scores.
    """

    def __init__(self, lambda_val, mu_val, base_score_function, graph_data, knn_edge=None, knn_weights=None):
        if lambda_val < 0 and lambda_val > 1:
            raise ValueError(
                "The parameter 'lambda_val' must be a value between 0 and 1.")
        if mu_val < 0 and mu_val > 1:
            raise ValueError(
                "The parameter 'mu_val' must be a value between 0 and 1.")
        if lambda_val + mu_val > 1:
            raise ValueError(
                "The summation of 'lambda_val' and 'mu_val' must not be greater than 1.")
        super(SNAPS, self).__init__(base_score_function, graph_data)

        self._lambda_val = lambda_val
        self._mu_val = mu_val
        self._knn_edge = knn_edge
        self._knn_weight = knn_weights

    def __call__(self, logits):
        base_scores = self._base_score_function(logits)

        if isinstance(self._edge_index, Tensor):
            if self._edge_weights is None:
                self._edge_weights = torch.ones(
                    self._edge_index.shape[1]).to(self._edge_index.device)
            adj = torch.sparse.FloatTensor(
                self._edge_index,
                self._edge_weights,
                (self._n_vertices, self._n_vertices))
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        elif isinstance(self._edge_index, SparseTensor):
            adj = self._edge_index
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        similarity_scores = 0.
        if self._knn_edge is not None:
            if isinstance(self._knn_edge, Tensor):
                if knn_weights is None:
                    knn_weights = torch.ones(
                        self._knn_edge.shape[1]).to(self._edge_index.device)
                adj_knn = torch.sparse.FloatTensor(
                    self._knn_edge,
                    knn_weights,
                    (self._n_vertices, self._n_vertices))
                knn_degs = torch.matmul(adj_knn, torch.ones(
                    (adj_knn.shape[0])).to(adj_knn.device))

            elif isinstance(self._knn_edge, SparseTensor):
                knn_degs = torch.matmul(adj_knn, torch.ones(
                    (adj_knn.shape[0])).to(adj_knn.device))

            similarity_scores = torch.linalg.matmul(
                adj_knn, base_scores) * (1 / (knn_degs + 1e-10))[:, None]

        neigh_scores = torch.linalg.matmul(
            adj, base_scores) * (1 / (degs + 1e-10))[:, None]

        scores = (1 - self._lambda_val - self._mu_val) * base_scores + \
            self._lambda_val * similarity_scores + \
            self._mu_val * neigh_scores

        return scores
