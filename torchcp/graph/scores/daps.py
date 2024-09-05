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
    :param n_vertices: the number of nodes.
    """

    def __init__(self, neigh_coef, n_vertices):
        if neigh_coef < 0 and neigh_coef > 1:
            raise ValueError("The parameter 'neigh_coef' must be a value between 0 and 1.")
        super(DAPS, self).__init__()

        self.__neigh_coef = neigh_coef
        self.__n_vertices = n_vertices

    def __call__(self, base_scores, edge_index, edge_weights=None):
        if isinstance(edge_index, Tensor):
            adj = torch.sparse.FloatTensor(
                edge_index,
                edge_weights,
                (self.__n_vertices, self.__n_vertices))
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        elif isinstance(edge_index, SparseTensor):
            adj = edge_index
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        diffusion_scores = torch.lianlg.matmul(adj, base_scores) * (1 / (degs + 1e-10))[:, None]

        scores = self.__neigh_coef * diffusion_scores + (1 - self.__neigh_coef) * base_scores

        return scores