# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/soroushzargar/DAPS

import torch
from torch import Tensor
from torch_geometric.typing import SparseTensor

from .base import BaseScore


class DAPS(BaseScore):
    """
    Diffusion Adaptive Prediction Sets (Zargarbashi et al., 2023)
    paper :https://proceedings.mlr.press/v202/h-zargarbashi23a/h-zargarbashi23a.pdf

    :param neigh_coef: the diffusion parameter which is a value in [0, 1].
    """

    def __init__(self, neigh_coef, base_score_function):
        if neigh_coef < 0 and neigh_coef > 1:
            raise ValueError(
                "The parameter 'neigh_coef' must be a value between 0 and 1.")
        super(DAPS, self).__init__(base_score_function)

        self.__neigh_coef = neigh_coef

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

        diffusion_scores = torch.linalg.matmul(
            adj, base_scores) * (1 / (degs + 1e-10))[:, None]

        scores = self.__neigh_coef * diffusion_scores + \
            (1 - self.__neigh_coef) * base_scores

        return scores
