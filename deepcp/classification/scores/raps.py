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
    def __init__(self, penalty, kreg, randomized=True):
        """
        :kreg : the rank of regularization [0,labels_num]
        """
        super(RAPS, self).__init__(randomized)
        self.__penalty = penalty
        self.__kreg = kreg

    def __call__(self, probs, y):

        # sorting probabilities
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = torch.where(indices == y)[0]
        reg = torch.maximum(self.__penalty * (idx + 1 - self.__kreg), torch.tensor(0))
        if not self.__randomized:
            return cumsum[idx] + reg
        else:
            U = torch.rand(1)[0]
            if idx == torch.tensor(0):
                return U * cumsum[idx] + reg
            else:
                return U * ordered[idx] + cumsum[idx - 1] + reg

    def predict(self, probs):
        I, ordered, cumsum = self._sort_sum(probs)
        U = torch.rand(probs.shape[0])
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[0] + 1) - self.__kreg), torch.tensor(0))
        if self.__randomized:
            ordered_scores = cumsum - ordered * U + reg
        else:
            ordered_scores = cumsum + reg
        return ordered_scores[torch.sort(I, descending=False)[1]]