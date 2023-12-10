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
    def __init__(self, weight, randomized=True):
        super(SAPS, self).__init__(randomized)
        if weight <= 0:
            raise ValueError("Weight must be a positive value.")
        self.__weight = weight


    def __call__(self, probs, y):
        
        # sorting probabilities
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = np.where(indices == y)[0]
        if not self._randomized:
            return self.__weight * idx + ordered[0]
        else:
            U = np.random.rand(1)
            if idx == 0:
                return U * cumsum[idx]
            else:
                return self.__weight * (idx - U) + ordered[0]

    def predict(self, probs):
        I, ordered, _ = self._sort_sum(probs)
        ordered[1:] = self.__weight
        cumsum = np.cumsum(ordered, axis=0)
        U = np.random.rand(probs.shape[0])
        if self._randomized:
            ordered_scores = cumsum - ordered * U
        else:
            ordered_scores = cumsum
        return ordered_scores[I.argsort(axis=0)]
