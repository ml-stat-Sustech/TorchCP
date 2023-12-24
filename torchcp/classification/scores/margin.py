# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from torchcp.classification.scores.aps import APS


class Margin(APS):

    def __init__(self, ) -> None:
        pass

        
    def _calculate_single_label(self, probs, y):
        row_indices = torch.arange(probs.size(0), device = probs.device)
        target_prob = probs[row_indices, y].clone()
        probs[row_indices, y] = -1
        second_highest_prob = torch.max(probs, dim=-1).values
        return second_highest_prob - target_prob
            

    def _calculate_all_label(self, probs):
        temp_probs = probs.unsqueeze(1).repeat(1, probs.shape[1], 1)
        indices = torch.arange(probs.shape[1]).to(probs.device)
        temp_probs[None, indices, indices] = torch.finfo(torch.float32).min
        scores = torch.max(temp_probs, dim=-1).values - probs
        return scores
    



