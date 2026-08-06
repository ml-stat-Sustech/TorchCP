# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch

from torchcp.regression.score.base import BaseScore


class BH_procedure(BaseScore):
    """
    Benjamini-Hochberg (BH) procedure:
        finds a p-value threshold from a list of p-values to determine which null hypotheses to reject, given a target
        FDR level 'alpha'.

    References:
        Paper: Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing
               (Benjamini and Hochberg, 1995)
        Link: https://www.jstor.org/stable/2346101
    """
    def __init__(self):
        super().__init__()

    def __call__(self, p_values, alpha):
        """
        Apply the Benjamini-Hochberg procedure.

        Args:
            p_values (torch.Tensor): A 1D tensor of p-values.
            alpha (float): The desired False Discovery Rate (FDR) level (e.g., 0.1).

        Returns:
            torch.Tensor: A 1D tensor of indices corresponding to the p-values (hypotheses) that are rejected.
        """
        p_values_sorted, _ = torch.sort(p_values)
        n_test = p_values_sorted.shape[0]

        k_range = torch.arange(1, n_test + 1, device=p_values_sorted.device)
        thresholds = k_range * alpha / n_test
        mask = p_values_sorted <= thresholds
        k_star = torch.max(torch.where(mask, k_range, torch.zeros_like(k_range))) if mask.any() else 0
        threshold = (k_star * alpha / n_test) if k_star > 0 else 0
        indices = torch.nonzero(p_values <= threshold, as_tuple=False).squeeze()

        return indices
