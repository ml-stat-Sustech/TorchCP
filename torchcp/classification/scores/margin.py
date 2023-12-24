# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from torchcp.classification.scores.base import BaseScoreFunction


class Margin(BaseScoreFunction):

    def __init__(self, ) -> None:
        """
        param score_type: either "softmax" "Identity", "log_softmax" or "log". Default: "softmax". A transformation for logits.
        """
        super().__init__()

        self.transform = lambda x: torch.softmax(x, dim=len(x.shape) - 1)

    def _compute_score(self, probs, index):
        
        pass

    def __call__(self, logits, y):
        probs = self.transform(logits)
        if len(logits.shape) == 1:
            target_prob = probs[y].clone()
            probs[y] = -1
            second_highest_prob = torch.max(probs, dim=-1).values
            return second_highest_prob - target_prob
        elif len(logits.shape) == 2:
            row_indices = torch.arange(probs.size(0), device = logits.device)
            target_prob = probs[row_indices, y].clone()
            probs[row_indices, y] = -1
            second_highest_prob = torch.max(probs, dim=-1).values
            return second_highest_prob - target_prob
        else:
            raise RuntimeError(" The dimension of logits must be less than 2.")
            


    def predict(self, logits):
        probs = self.transform(logits)
        
        if len(probs.shape) == 1:
            temp_probs = probs.repeat(logits.shape[0], 1)
            indices = torch.arange(logits.shape[0]).to(logits.device)
            temp_probs[indices, indices] = torch.finfo(torch.float32).min
            scores = torch.max(temp_probs, dim=-1).values - probs
        elif len(probs.shape) == 2:
            temp_probs = probs.unsqueeze(1).repeat(1, probs.shape[1], 1)
            indices = torch.arange(probs.shape[1]).to(logits.device)
            temp_probs[None, indices, indices] = torch.finfo(torch.float32).min
            scores = torch.max(temp_probs, dim=-1).values - probs
        else:
            raise RuntimeError(" The dimension of logits must be less than 2.")
        return scores
    



