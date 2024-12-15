# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .base import BaseScore
from torchcp.graph.utils import compute_adj_knn


class SNAPS(BaseScore):
    """
    Method: Similarity-Navigated Adaptive Prediction Sets
    Paper: Similarity-Navigated Conformal Prediction for Graph Neural Networks (Song et al., 2024)
    Link: https://arxiv.org/pdf/2405.14303
    Github:

    Parameters:
        xi (float): 
            The weight parameter for neighborhood-based scores, where 0 <= xi <= 1.

        mu (float): 
            The weight parameter for similarity-based scores, where 0 <= mu <= 1.

        knn_edge (torch.Tensor, optional): 
            An edge list representing the k-nearest neighbors (k-NN) for each node. It may be constructed based on
            the similarity of nodes' feature. The shape is (2, E), where E is the number of edges in the kNN graph.
            The first row contains the source node indices, and the second row contains the target node indices.

        knn_weight (torch.Tensor, optional): 
            The weights associated with each k-NN edge, if applicable. Defaults to uniform weights.
    """

    def __init__(self, 
                 graph_data, 
                 base_score_function, 
                 xi=1 / 3, 
                 mu=1 / 3, 
                 knn_edge=None, 
                 knn_weight=None,
                 features=None, 
                 k=20):
        super(SNAPS, self).__init__(graph_data, base_score_function)
        if xi < 0 or xi > 1:
            raise ValueError(
                "The parameter 'xi' must be a value between 0 and 1.")
        if mu < 0 or mu > 1:
            raise ValueError(
                "The parameter 'mu' must be a value between 0 and 1.")
        if xi + mu > 1:
            raise ValueError(
                "The summation of 'xi' and 'mu' must not be greater than 1.")

        self._xi = xi
        self._mu = mu

        if features is not None:
            knn_edge, knn_weight = compute_adj_knn(features, k)

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

    def __call__(self, logits, labels=None):
        base_scores = self._base_score_function(logits)

        similarity_scores = 0.
        if self._adj_knn is not None:
            similarity_scores = torch.linalg.matmul(
                self._adj_knn, base_scores) * (1 / (self._knn_degs + 1e-10))[:, None]

        neigh_scores = torch.linalg.matmul(
            self._adj, base_scores) * (1 / (self._degs + 1e-10))[:, None]

        scores = (1 - self._xi - self._mu) * base_scores + \
                 self._xi * similarity_scores + \
                 self._mu * neigh_scores

        if labels is None:
            return scores
        else:
            return scores[torch.arange(scores.shape[0]), labels]
