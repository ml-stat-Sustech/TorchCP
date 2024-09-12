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

    def __init__(self, lambda_val, mu_val, base_score_function):
        if lambda_val < 0 and lambda_val > 1:
            raise ValueError(
                "The parameter 'lambda_val' must be a value between 0 and 1.")
        if mu_val < 0 and mu_val > 1:
            raise ValueError(
                "The parameter 'mu_val' must be a value between 0 and 1.")
        if lambda_val + mu_val > 1:
            raise ValueError(
                "The summation of 'lambda_val' and 'mu_val' must not be greater than 1.")
        super(SNAPS, self).__init__(base_score_function)

        self.__lambda_val = lambda_val
        self.__mu_val = mu_val

    def __call__(self, logits, n_vertices, edge_index, edge_weights=None, knn_edge=None, knn_weights=None):
        base_scores = self._base_score_function(logits)
        if isinstance(edge_index, Tensor):
            if edge_weights is None:
                edge_weights = torch.ones(
                    edge_index.shape[1]).to(edge_index.device)
            adj = torch.sparse.FloatTensor(
                edge_index,
                edge_weights,
                (n_vertices, n_vertices))
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        elif isinstance(edge_index, SparseTensor):
            adj = edge_index
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        similarity_scores = 0.
        if knn_edge is not None:
            if isinstance(knn_edge, Tensor):
                if knn_weights is None:
                    knn_weights = torch.ones(
                        knn_edge.shape[1]).to(edge_index.device)
                adj_knn = torch.sparse.FloatTensor(
                    knn_edge,
                    knn_weights,
                    (n_vertices, n_vertices))
                knn_degs = torch.matmul(adj_knn, torch.ones(
                    (adj_knn.shape[0])).to(adj_knn.device))

            elif isinstance(knn_edge, SparseTensor):
                knn_degs = torch.matmul(adj_knn, torch.ones(
                    (adj_knn.shape[0])).to(adj_knn.device))

            similarity_scores = torch.linalg.matmul(
                adj_knn, base_scores) * (1 / (knn_degs + 1e-10))[:, None]

        neigh_scores = torch.linalg.matmul(
            adj, base_scores) * (1 / (degs + 1e-10))[:, None]

        scores = (1 - self.__lambda_val - self.__mu_val) * base_scores + \
            self.__lambda_val * similarity_scores + \
            self.__mu_val * neigh_scores

        return scores
