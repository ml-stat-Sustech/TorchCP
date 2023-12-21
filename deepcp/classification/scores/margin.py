# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from deepcp.classification.scores.base import BaseScoreFunction


class Margin(BaseScoreFunction):

    def __init__(self, ) -> None:
        """
        param score_type: either "softmax" "Identity", "log_softmax" or "log". Default: "softmax". A transformation for logits.
        """
        super().__init__()

        self.transform = lambda x: torch.softmax(x, dim=len(x.shape) - 1)

    def _compute_score(self, probs, index):
        target_prob = probs[index].clone()
        probs[index] = -1
        second_highest_prob = torch.max(probs, dim=-1).values
        return second_highest_prob - target_prob

    def __call__(self, logits, y):
        probs = self.transform(logits)
        if len(logits.shape) > 1:
            scores = torch.zeros(logits.shape[0]).to(logits.device)
            for i in range(logits.shape[0]):
                scores[i] = self._compute_score(probs[i], y[i])
            return scores
        else:
            return self._compute_score(probs, y)

    def predict(self, logits):
        probs = self.transform(logits)
        temp_probs = probs.repeat(logits.shape[0], 1)
        indices = torch.arange(logits.shape[0])
        temp_probs[indices, indices] = torch.finfo(torch.float32).min
        return torch.max(temp_probs, dim=1).values - probs
