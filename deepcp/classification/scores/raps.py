# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import torch

from deepcp.classification.scores.aps import APS


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
            raise ValueError("param 'penalty' must be a positive value.")
        if kreg < 0:
            raise ValueError("param 'kreg' must be a nonnegative  value.")
        if type(kreg) != int:
            raise TypeError("param 'kreg' must be a integer.")
        super(RAPS, self).__init__()
        self.__penalty = penalty
        self.__kreg = kreg

    def __call__(self, logits, y):
        probs = self.transform(logits)

        # sorting probabilities
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = torch.where(indices == y)[0][0]

        reg = torch.maximum(self.__penalty * (idx + 1 - self.__kreg), torch.tensor(0).to(logits.device))
        U = torch.rand(1).to(logits.device)
        if idx == torch.tensor(0).to(logits.device):
            return U * cumsum[idx] + reg
        else:
            return U * ordered[idx] + cumsum[idx - 1] + reg

    def predict(self, logits):
        probs = self.transform(logits)
        I, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape).to(logits.device)
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1).to(logits.device) - self.__kreg), torch.tensor(0).to(logits.device))
        ordered_scores = cumsum - ordered * U + reg
        return ordered_scores[torch.sort(I, descending=False, dim=-1)[1]]
