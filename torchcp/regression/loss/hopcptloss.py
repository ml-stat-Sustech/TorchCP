# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

__all__ = ["HopCPTLoss"]


class HopCPTLoss(nn.Module):
    r"""
    HopCPT Loss for learning with correlated predictions.

    This loss function is designed to improve the accuracy of predictions 
    when correlations exist between different prediction tasks. 
    It encourages the model to learn to predict errors in a way that 
    accounts for these correlations.


    Shape:
        - Input: 
            - `errors`: A tensor of shape `(batch_size, num_tasks)` 
                      representing the prediction errors for each task.
            - `association_matrix`: A tensor of shape `(num_tasks, num_tasks)` 
                      representing the pairwise association strengths between tasks. 
        - Output: A scalar representing the HopCPT loss.

    Reference:
        Paper: Conformal Prediction for Time Series with Modern Hopfield Networks (Auer, et al., 2023)
        Link: https://openreview.net/forum?id=KTRwpWCMsC
        Github: https://github.com/ml-jku/HopCPT
    
    The loss function is defined as:

    .. math::
        \mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( |e_i| - \sum_{j=1}^{M} a_{ij} e_j \right)^2

    where:
        - `N` is the batch size.
        - `M` is the number of tasks.
        - `e_i` is the prediction error for task `i`.
        - `a_{ij}` is the association strength between tasks `i` and `j`.

    This loss encourages the model to learn to predict errors that are 
    consistent with the observed correlations between tasks.

    Examples::

        >>> loss_fn = HopCPTLoss()
        >>> errors = torch.randn(10, 5)  # Batch size of 10, 5 tasks
        >>> association_matrix = torch.rand(5, 5)  # Association matrix
        >>> loss = loss_fn(errors, association_matrix)
    """
    def __init__(self):
        super(HopCPTLoss, self).__init__()

    def forward(self, errors, association_matrix):
        """
        Computes the HopCPT loss.

        Args:
            errors (torch.Tensor): A tensor of shape `(batch_size, num_tasks)` 
                      representing the prediction errors for each task.
            association_matrix (torch.Tensor): A tensor of shape `(num_tasks, num_tasks)` 
                      representing the pairwise association strengths between tasks.

        Returns:
            torch.Tensor: The computed HopCPT loss.
        """
        weighted_errors = torch.matmul(association_matrix, errors.unsqueeze(-1)).squeeze(-1) 
        loss = torch.mean((torch.abs(errors) - weighted_errors) ** 2) 
        return loss
