# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import torch

from torchcp.classification.scores.aps import APS


class RAPS(APS):
    """
    Regularized Adaptive Prediction Sets (Angelopoulos et al., 2020)
    paper : https://arxiv.org/abs/2009.14193
    """

    def __init__(self, penalty, kreg=0):
        """
        when penalty = 0, RAPS=APS.

        :param kreg : the rank of regularization which is an integer in [0,labels_num].
        """
        if penalty <= 0:
            raise ValueError("The parameter 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("The parameter 'kreg' must be a nonnegative value.")
        if type(kreg) != int:
            raise TypeError("The parameter 'kreg' must be a integer.")
        super(RAPS, self).__init__()
        self.__penalty = penalty
        self.__kreg = kreg

    def __call__(self, logits, y):
        assert len(logits.shape) <= 2, "The dimension of logits must be less than 2."
        if len(logits) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        # sorting probabilities
        indices, ordered, cumsum = self._sort_sum(probs)
        return self.__compute_score(indices, y, cumsum, ordered)
        
        

    def predict(self, logits):
        probs = torch.softmax(logits, dim=-1)
        I, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape, device = logits.device)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=logits.device) - self.__kreg),torch.tensor(0).to(logits.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(I, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores
    

    
    def __compute_score(self, indices, y, cumsum, ordered):        
        U = torch.rand(indices.shape[0], device = indices.device)
        idx = torch.where(indices == y.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(indices.device))
        scores_first_rank  = U * ordered[idx] + reg
        idx_minus_one = (idx[0], idx[1] - 1) 
        scores_usual  = U * ordered[idx] + cumsum[idx_minus_one] + reg
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)

