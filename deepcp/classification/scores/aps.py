# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import torch

from deepcp.classification.scores.base import BaseScoreFunction


class APS(BaseScoreFunction):
    """
    Adaptive Prediction Sets (Romano et al., 2020)
    paper :https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf
    """

    def __init__(self, ):
        super(APS, self).__init__()
        self.transform = lambda x: torch.softmax(x, dim=len(x.shape) - 1)

    def __call__(self, logits, y):
        probs = self.transform(logits)
        indices, ordered, cumsum = self._sort_sum(probs)
        if len(probs.shape) == 1:
            return self._compute_score(indices, y, cumsum, ordered)
        else:
            scores = torch.zeros(probs.shape[0]).to(logits.device)
            for i in range(probs.shape[0]):
                scores[i] = self._compute_score(indices[i, :], y[i], cumsum[i, :], ordered[i, :])
            return scores

    def predict(self, logits):
        probs = self.transform(logits)
        I, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape)
        ordered_scores = cumsum - ordered * U
        return ordered_scores[torch.sort(I, descending=False, dim=-1)[1]]

    def _sort_sum(self, probs):
        # ordered: the ordered probabilities in descending order
        # indices: the rank of ordered probabilities in descending order
        # cumsum: the accumulation of sorted probabilities
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _compute_score(self, indices, y, cumsum, ordered):
        idx = torch.where(indices == y)[0][0]
        U = torch.rand(1).to(indices.device)
        if idx == torch.tensor(0).to(indices.device):
            return U * cumsum[idx]
        else:
            return U * ordered[idx] + cumsum[idx - 1]
