# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import torch

from .aps import APS


class RAPS(APS):
    """
    Method: Regularized Adaptive Prediction Sets 
    Paper: Uncertainty Sets for Image Classifiers using Conformal Prediction (Angelopoulos et al., 2020)
    Link: https://arxiv.org/abs/2009.14193
    
    Args:
        penalty (float): The weight of regularization. When penalty = 0, RAPS=APS.
        kreg (int, optional): The rank of regularization which is an integer in [0, labels_num]. Default is 0.
        score_type (str, optional): The type of score to use. Default is "softmax".
        randomized (bool, optional): Whether to use randomized scores. Default is True.
        
    Examples::
        >>> raps = RAPS(penalty=0.1, kreg=1, score_type="softmax", randomized=True)
        >>> probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
        >>> scores_all = raps._calculate_all_label(probs)
        >>> print(scores_all)
        >>> scores_single = raps._calculate_single_label(probs, torch.tensor([2, 1]))
        >>> print(scores_single)
        
    """

    def __init__(self, score_type="softmax", randomized=True, penalty=0, kreg=0):

        super().__init__(score_type=score_type, randomized=randomized)
        if penalty < 0:
            raise ValueError("The parameter 'penalty' must be a nonnegative value.")

        if type(kreg) != int or kreg < 0:
            raise TypeError("The parameter 'kreg' must be a nonnegative integer.")
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
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
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        idx_minus_one = (idx[0], idx[1] - 1)
        scores = cumsum[idx_minus_one] - U * ordered[idx] + reg
        return scores
