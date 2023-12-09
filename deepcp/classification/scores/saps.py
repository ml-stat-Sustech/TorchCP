# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


# The reference repository is https://arxiv.org/abs/2310.06430


import numpy as np
import torch

from deepcp.classification.scores.aps import APS


class SAPS(APS):
    def __init__(self, penalty=0, randomized=True):
        super(SAPS, self).__init__(randomized)
        self.__penalty = penalty
        #todo: add lambda?


    def __call__(self, probs, y):
        # sorting probabilities
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = torch.where(indices == y)[0]
        if not self.__randomized:
            return self.__penalty * idx + ordered[0]
        else:
            U = torch.rand(1)[0]
            if idx == torch.tensor(0):
                return U * cumsum[idx]
            else:
                return self.__penalty * (idx - U) + ordered[0]

    def predict(self, probs):
        I, ordered, _ = self._sort_sum(probs)
        ordered[1:] = self.__penalty
        cumsum = torch.cumsum(ordered, dim=0)
        U = torch.rand(probs.shape[0])
        if self.__randomized:
            ordered_scores = cumsum - ordered * U
        else:
            ordered_scores = cumsum
        return ordered_scores[torch.sort(I, descending=False)[1]]
