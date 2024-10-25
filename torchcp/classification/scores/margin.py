# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from .aps import APS


class Margin(APS):
    """
    Method: Margin non-conformity score
    Paper: Bias reduction through conditional conformal prediction (Löfström et al., 2015)
    Link:https://dl.acm.org/doi/abs/10.3233/IDA-150786
    """

    def __init__(self, score_type="softmax"):
        super().__init__(score_type)

    def _calculate_single_label(self, probs, label):
        row_indices = torch.arange(probs.size(0), device=probs.device)
        target_prob = probs[row_indices, label].clone()
        probs[row_indices, label] = -1

        # the largest probs except for the correct labels
        largest_probs_ex_correct_labels = torch.max(probs, dim=-1).values
        return largest_probs_ex_correct_labels - target_prob

    def _calculate_all_label(self, probs):
        batch_size, num_labels = probs.shape
            
        values, indices = torch.topk(probs, k=2, dim=1)
    
        max_values = values[:, 0].unsqueeze(1).expand(-1, num_labels)
        second_max_values = values[:, 1].unsqueeze(1).expand(-1, num_labels)
        max_indices = indices[:, 0].unsqueeze(1).expand(-1, num_labels)
        position_indices = torch.arange(num_labels).expand(batch_size, -1).to(probs.device)
        
        selected_values = torch.where(position_indices == max_indices, 
                                    second_max_values, 
                                    max_values)
        
        scores = selected_values - probs
        return scores
