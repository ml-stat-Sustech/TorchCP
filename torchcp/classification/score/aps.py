# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.score.thr import THR


class APS(THR):
    """
    Method: Adaptive Prediction Sets (APS)
    Paper: Classification with Valid and Adaptive Coverage (Romano et al., 2020)
    Link:https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf
    
    Args:
        score_type (str, optional): The type of score to use. Default is "softmax".
        randomized (bool, optional): Whether to use randomized scores. Default is True.

    Methods:
        _calculate_all_label(probs):
            Calculate non-conformity scores for all classes.
        _sort_sum(probs):
            Sort probabilities and calculate cumulative sum.
        _calculate_single_label(probs, label):
            Calculate non-conformity score for the ground-truth label.
    
    Examples::
        >>> aps = APS(score_type="softmax", randomized=True)
        >>> probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
        >>> scores = aps._calculate_all_label(probs)
        >>> print(scores)
    """

    def __init__(self, score_type="softmax", randomized=True):
        super().__init__(score_type)
        self.randomized = randomized

    def _calculate_all_label(self, probs):
        """
        Calculate non-conformity scores for all labels.

        Args:
            probs (torch.Tensor): The prediction probabilities.

        Returns:
            torch.Tensor: The non-conformity scores.
        """
        if probs.dim() == 1 or probs.dim() > 2:
            raise ValueError("Input probabilities must be 2D.")
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs)

        ordered_scores = cumsum - ordered * U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _sort_sum(self, probs):
        """
        Sort probabilities and calculate cumulative sum.

        Args:
            probs (torch.Tensor): The prediction probabilities.

        Returns:
            tuple: A tuple containing:
                - indices (torch.Tensor): The rank of ordered probabilities in descending order.
                - ordered (torch.Tensor): The ordered probabilities in descending order.
                - cumsum (torch.Tensor): The accumulation of sorted probabilities.
        """
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label):
        """
        Calculate non-conformity score for a single label.

        Args:
            probs (torch.Tensor): The prediction probabilities.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The non-conformity score for the given label.
        """
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(indices.shape[0], device=probs.device)
        else:
            U = torch.zeros(indices.shape[0], device=probs.device)

        idx = torch.where(indices == label.view(-1, 1))
        scores = cumsum[idx] - U * ordered[idx]
        return scores
