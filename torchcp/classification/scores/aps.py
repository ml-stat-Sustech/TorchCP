# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import torch

from torchcp.classification.scores.base import BaseScore


class APS(BaseScore):
    """
    Adaptive Prediction Sets (Romano et al., 2020)
    paper :https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf
    """

    def __init__(self):
        super(APS, self).__init__()

    def __call__(self, logits, y):
        assert len(logits.shape) <= 2, "The dimension of logits must be less than 2."
        if len(logits) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        indices, ordered, cumsum = self._sort_sum(probs)
        return self.__compute_score(indices, y, cumsum, ordered)

    def predict(self, logits):
        probs = torch.softmax(logits, dim=-1)
        I, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=logits.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(I, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        # ordered: the ordered probabilities in descending order
        # indices: the rank of ordered probabilities in descending order
        # cumsum: the accumulation of sorted probabilities
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def __compute_score(self, indices, y, cumsum, ordered):
        U = torch.rand(indices.shape[0], device = indices.device)
        idx = torch.where(indices == y.view(-1, 1))
        scores_first_rank  = U * cumsum[idx]
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual  = U * ordered[idx] + cumsum[idx_minus_one]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
            

