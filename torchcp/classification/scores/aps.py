# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import torch

from .base import BaseScore


class APS(BaseScore):
    """
    Adaptive Prediction Sets (Romano et al., 2020)
    paper :https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf
    """

    def __call__(self, logits, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        if label is None:
            return self._calculate_all_label(probs)
        else:
            return self._calculate_single_label(probs, label)

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device=probs.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        # ordered: the ordered probabilities in descending order
        # indices: the rank of ordered probabilities in descending order
        # cumsum: the accumulation of sorted probabilities
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        idx_minus_one = (idx[0], idx[1] - 1)
        scores_usual = U * ordered[idx] + cumsum[idx_minus_one]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
