# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch

from torchcp.classification.scores.aps import APS


class SAPS(APS):
    """
    Sorted Adaptive Prediction Sets (Huang et al., 2023)
    paper: https://arxiv.org/abs/2310.06430
    """

    def __init__(self, weight):
        """
        :param weight: the weigth of label ranking.
        """
        super(SAPS, self).__init__()
        if weight <= 0:
            raise ValueError("param 'weight' must be a positive value.")
        self.__weight = weight

    def __call__(self, logits, y):
        probs = self.transform(logits)
        # sorting probabilities
        indices, ordered, cumsum = self._sort_sum(probs)
        idx = torch.where(indices == y)[0][0]
        U = torch.rand(1).to(logits.device)
        if idx == torch.tensor(0):
            return U * cumsum[idx]
        else:
            return self.__weight * (idx - U) + ordered[0]

    def predict(self, logits):
        probs = self.transform(logits)
        I, ordered, _ = self._sort_sum(probs)
        ordered[1:] = self.__weight
        cumsum = torch.cumsum(ordered, dim=-1)
        U = torch.rand(probs.shape[0]).to(logits.device)
        ordered_scores = cumsum - ordered * U

        return ordered_scores[torch.sort(I, descending=False, dim=-1)[1]]
