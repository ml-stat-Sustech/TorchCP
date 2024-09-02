# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch

from .aps import APS


class SAPS(APS):
    """
    Sorted Adaptive Prediction Sets (Huang et al., 2023)
    paper: https://arxiv.org/abs/2310.06430
    
    :param weight: the weight of label ranking.
    """

    def __init__(self, weight, score_type="softmax"):
        super().__init__(score_type)
        if weight <= 0:
            raise ValueError("The parameter 'weight' must be a positive value.")
        self.__weight = weight

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        ordered[:, 1:] = self.__weight
        cumsum = torch.cumsum(ordered, dim=-1)
        U = torch.rand(probs.shape, device=probs.device)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        scores_usual = self.__weight * (idx[1] - U) + ordered[:, 0]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
