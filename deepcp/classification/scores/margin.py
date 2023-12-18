# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import scipy
import torch

from deepcp.classification.scores.base import BaseScoreFunction


class Margin(BaseScoreFunction):
    """
    Threshold conformal predictor (Sadinle et al., 2016)
    paper : https://arxiv.org/abs/1609.00451
    """

    def __init__(self,) -> None:
        """
        param score_type: either "softmax" "Identity", "log_softmax" or "log". Default: "softmax". A transformation for logits.
        """
        super().__init__()

        self.transform = lambda x: torch.softmax(x,dim= len(x.shape)-1)


    def __call__(self, logits, y):
        probs = self.transform(logits)
        if len(logits.shape) >1:
            return 1 - self.transform(logits)[torch.arange(y.shape[0]),y]
        else:
            target_prob = probs[y]
            second_highest_prob = torch.max(
                torch.cat((probs[:y], probs[y + 1:])))

            return second_highest_prob - target_prob

    def predict(self, logits):
        probs = self.transform(logits)
        scores = torch.zeros_like(probs)
        for i in range(probs.shape[0]):
            target_prob = probs[i]
            second_highest_prob = torch.max(
                torch.cat((probs[:i], probs[i + 1:])))
            scores[i] = second_highest_prob - target_prob
        return scores
