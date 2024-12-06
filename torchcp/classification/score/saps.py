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
    Method: Sorted Adaptive Prediction Sets 
    Paper: Conformal Prediction for Deep Classifier via Label Ranking (Huang et al., 2023)
    Link: https://arxiv.org/abs/2310.06430
    Github: https://github.com/ml-stat-Sustech/conformal_prediction_via_label_ranking
    
    Args:
        weight (float): The weight of label ranking. Must be a positive value.
        score_type (str, optional): The type of score to use. Default is "softmax".
        randomized (bool, optional): Whether to use randomized scores. Default is True.
    
    Examples::
        >>> saps = SAPS(weight=0.5, score_type="softmax", randomized=True)
        >>> probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
        >>> scores_all = saps._calculate_all_label(probs)
        >>> print(scores_all)
        >>> scores_single = saps._calculate_single_label(probs, torch.tensor([2, 1]))
        >>> print(scores_single)
    """

    def __init__(self, score_type="softmax", randomized=True, weight=0.2):
        super().__init__(score_type, randomized)
        if weight <= 0:
            raise ValueError("The parameter 'weight' must be a positive value.")
        if not isinstance(randomized, bool):
            raise ValueError("The parameter 'randomized' must be a boolean.")
        self.__weight = weight

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        ordered[:, 1:] = self.__weight
        cumsum = torch.cumsum(ordered, dim=-1)
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs)
        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label):
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(indices.shape[0], device=probs.device)
        else:
            U = torch.zeros(indices.shape[0], device=probs.device)
        idx = torch.where(indices == label.view(-1, 1))
        scores_first_rank = U * cumsum[idx]
        scores_usual = self.__weight * (idx[1] - U) + ordered[:, 0]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
