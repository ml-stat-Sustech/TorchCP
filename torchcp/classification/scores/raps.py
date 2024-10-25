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
    
    :param penalty: the weight of regularization. When penalty = 0, RAPS=APS.
    :param kreg: the rank of regularization which is an integer in [0,labels_num].
    """

    def __init__(self, penalty, kreg=0, score_type="softmax", randomized=True):
        super().__init__(score_type, randomized)
        if penalty <= 0:
            raise ValueError("The parameter 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("The parameter 'kreg' must be a nonnegative value.")
        if type(kreg) != int:
            raise TypeError("The parameter 'kreg' must be a integer.")
        super(RAPS, self).__init__()
        self.__penalty = penalty
        self.__kreg = kreg

    def _calculate_all_label(self, probs):
        indices, ordered, cumsum = self._sort_sum(probs)
        if self.randomized:
            U = torch.rand(probs.shape, device=probs.device)
        else:
            U = torch.zeros_like(probs.shape)
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
        # indices, ordered, cumsum = self._sort_sum(probs)
        # idx = torch.where(indices == label.view(-1, 1))
        # reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        # return super()._calculate_single_label(probs, label) + reg
