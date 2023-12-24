# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from torchcp.classification.scores.base import BaseScore


class Margin(BaseScore):

    def __init__(self, ) -> None:
        """
        param score_type: either "softmax" "Identity", "log_softmax" or "log". Default: "softmax". A transformation for logits.
        """
        super().__init__()


    def __call__(self, logits, y):
        assert len(logits.shape) <= 2, "The dimension of logits must be less than 2."
        if len(logits) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)

        row_indices = torch.arange(probs.size(0), device = logits.device)
        target_prob = probs[row_indices, y].clone()
        probs[row_indices, y] = -1
        second_highest_prob = torch.max(probs, dim=-1).values
        return second_highest_prob - target_prob
            

    def predict(self, logits):
        assert len(logits.shape) <= 2, "The dimension of logits must be less than 2."
        if len(logits) == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)
        temp_probs = probs.unsqueeze(1).repeat(1, probs.shape[1], 1)
        indices = torch.arange(probs.shape[1]).to(logits.device)
        temp_probs[None, indices, indices] = torch.finfo(torch.float32).min
        scores = torch.max(temp_probs, dim=-1).values - probs
        return scores
    



