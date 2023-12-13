# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# The reference repository is https://github.com/aangelopoulos/conformal_classification


import numpy as np

from deepcp.classification.scores.base import BaseScoreFunction


class APS(BaseScoreFunction):
    def __init__(self,):
        super(APS, self).__init__()

    def __call__(self, probs, y):

        # sorting probabilities
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = np.where(indices == y)[0]
        
        U = np.random.rand()
        if idx == np.array(0):
            return U * cumsum[idx]
        else:
            return U * ordered[idx] + cumsum[idx - 1]

    def predict(self, probs):
        I, ordered, cumsum = self._sort_sum(probs)
        U = np.random.rand(probs.shape[0])

        ordered_scores = cumsum - ordered * U

        return ordered_scores[I.argsort(axis=0)]

    def _sort_sum(self, probs):
        # indices: the rank of ordered probabilities in descending order
        indices = probs.argsort(axis=0)[::-1]
        # ordered: the ordered probabilities in descending order
        ordered = np.sort(probs,axis=0)[::-1]
        # the accumulation of sorted probabilities
        cumsum = np.cumsum(ordered,axis=0) 
        return indices, ordered, cumsum
