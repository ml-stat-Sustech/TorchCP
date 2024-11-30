# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import torch

from .thr import THR


class TOPK(THR):
    """
    Method: TOPK prediction sets
    Paper: Uncertainty Sets for Image Classifiers using Conformal Prediction (Angelopoulos et al., 2020)
    Link: https://arxiv.org/abs/2009.14193
    
     Args:
        score_type (str, optional): The type of score to use. Default is "softmax".
        randomized (bool, optional): Whether to use randomized scores. Default is True.

    Examples::
        >>> topk = TOPK(score_type="softmax", randomized=True)
        >>> probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
        >>> scores = topk._calculate_all_label(probs)
        >>> print(scores)

    """

    def __init__(self, randomized = True, score_type="softmax"):
        super().__init__(score_type)
        self.randomized = randomized
        
        
    def _calculate_all_label(self, probs):
        """Calculate scores for all labels using binary values
        
        Args:
            probs (torch.Tensor): the prediction probabilities
            
        Returns:
            torch.Tensor: the non-conformity scores
        """
        # Get the ordering information from original probs
        indices, _, cumsum = self._sort_sum(probs)
            
        
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs)
            
        ordered_scores = cumsum - U
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        
        return scores
    
    def _sort_sum(self, probs):
        """Sort values and return indices and cumulative sums
        """
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        ones = torch.ones_like(ordered)
        cumsum = torch.cumsum(ones, dim=-1)
        return indices, ones, cumsum
        
    def _calculate_single_label(self, probs, label):
        """Calculate score for a single label
        
        Args:
            probs (torch.Tensor): the prediction probabilities
            label (torch.Tensor): the true label
            
        Returns:
            torch.Tensor: the non-conformity scores
        """
        indices, ones, cumsum = self._sort_sum(probs)
        
        if self.randomized:
            U = torch.rand(indices.shape[0], device=probs.device)
        else:
            U = torch.zeros(indices.shape[0], device=probs.device)
            
        idx = torch.where(indices == label.view(-1, 1))
        scores = cumsum[idx] - U
        
        return scores
    

