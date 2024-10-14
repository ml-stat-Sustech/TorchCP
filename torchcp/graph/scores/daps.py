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
    Method: Diffusion Adaptive Prediction Sets
    Paper: Conformal Prediction Sets for Graph Neural Networks (Zargarbashi et al., 2023)
    Link: https://proceedings.mlr.press/v202/h-zargarbashi23a/h-zargarbashi23a.pdf
    Github: https://github.com/soroushzargar/DAPS

    :param neigh_coef: the diffusion parameter which is a value in [0, 1].
    """

    def __init__(self, neigh_coef, base_score_function, graph_data):
        super(DAPS, self).__init__(base_score_function, graph_data)
        if neigh_coef < 0 and neigh_coef > 1:
            raise ValueError(
                "The parameter 'neigh_coef' must be a value between 0 and 1.")

        self._neigh_coef = neigh_coef

    def __call__(self, logits):
        base_scores = self._base_score_function(logits)

        if isinstance(self._edge_index, Tensor):
            if self._edge_weight is None:
                self._edge_weight = torch.ones(
                    self._edge_index.shape[1]).to(self._edge_index.device)
            adj = torch.sparse.FloatTensor(
                self._edge_index,
                self._edge_weight,
                (self._n_vertices, self._n_vertices))
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        elif isinstance(self._edge_index, SparseTensor):
            adj = self._edge_index
            degs = torch.matmul(adj, torch.ones((adj.shape[0])).to(adj.device))

        diffusion_scores = torch.linalg.matmul(
            adj, base_scores) * (1 / (degs + 1e-10))[:, None]

        scores = self._neigh_coef * diffusion_scores + \
            (1 - self._neigh_coef) * base_scores

        return scores
