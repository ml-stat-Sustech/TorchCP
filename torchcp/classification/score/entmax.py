# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.score.base import BaseScore


class EntmaxScore(BaseScore):
    """
    Score functions based on gamma-entmax transformations as described in
    'Sparse Activations as Conformal Predictors' (Campos et al., AISTATS 2025).

    Args:
        gamma (float, optional): The gamma parameter for entmax transformation.
            - gamma = 1: softmax with log-margin score
            - gamma = 2: sparsemax
            - gamma > 1: sparse entmax
            Defaults to 2.0 (sparsemax).

    Attributes:
        gamma (float): The gamma parameter for entmax.
        temperature (float): The temperature scaling factor.
        
    Examples::
        >>> entmax = EntmaxScore(gamma=2.0, temperature=1.0)  # Sparsemax
        >>> logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
        >>> scores_all = entmax(logits)
        
        >>> # Using gamma=1 (softmax with log-margin)
        >>> entmax = EntmaxScore(gamma=1.0, temperature=0.5)
        >>> scores_custom = entmax(logits, label=torch.tensor([1, 0]))
        
    References:
        Campos, M. M. et al., (2025). Sparse Activations as Conformal Predictors.
        Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS).
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        
        if gamma < 1.0:
            raise ValueError("Gamma must be >= 1.0.")
        
        self.gamma = gamma
        if self.gamma == 1.0:
            self.delta = 1
        else:
            self.delta = 1 / (self.gamma - 1)

    def __call__(self, logits, label=None):
        """
        Calculate non-conformity scores based on gamma-entmax.

        Args:
            logits (torch.Tensor): The logits output from the model.
            label (torch.Tensor, optional): The ground truth label. Default is None.

        Returns:
            torch.Tensor: The non-conformity scores.
        """
        if len(logits.shape) > 2:
            raise ValueError("Dimension of logits must be at most 2.")
        
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        
        
        if label is None:
            return self._calculate_all_label(logits)
        else:
            return self._calculate_single_label(logits, label)

    def _calculate_single_label(self, logits, label):
        """
        Calculate non-conformity score for a single label based on paper's definition.

        Args:
            logits (torch.Tensor): Scaled logits.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The non-conformity score for the given label.
        """
        sorted_logits, indices = torch.sort(logits, dim=-1, descending=True)
        batch_size = logits.shape[0]
        label_ranks = torch.zeros(batch_size, device=logits.device, dtype=torch.long)
        
        for i in range(batch_size):
            label_ranks[i] = (indices[i] == label[i]).nonzero(as_tuple=True)[0]
        
        if self.gamma == 1.0:  # Softmax case (log-margin)
            scores = sorted_logits[:, 0] - logits[torch.arange(batch_size), label]  # z_1 - z_{k(y)}
            return scores
        elif self.gamma == 2.0:  # Sparsemax case
            scores = torch.zeros(batch_size, device=logits.device)
            for i in range(batch_size):
                k = label_ranks[i]
                if k > 0:
                    scores[i] = torch.sum(sorted_logits[i, :k] - sorted_logits[i, k])
            return scores
        else:  # General gamma-entmax case
            scores = torch.zeros(batch_size, device=logits.device)
            for i in range(batch_size):
                k = label_ranks[i]
                if k > 0:
                    scores[i] = self._calculate_score_logsumexp(sorted_logits[i, :k] - sorted_logits[i, k])
            return scores

    def _calculate_all_label(self, logits):
        """
        Calculate non-conformity scores for all labels.

        Args:
            logits (torch.Tensor): Scaled logits.

        Returns:
            torch.Tensor: The non-conformity scores for all labels.
        """
        sorted_logits, indices = torch.sort(logits, dim=-1, descending=True)
        batch_size, num_classes = logits.shape
        
        if self.gamma == 1.0:  # Softmax case (log-margin)
            scores = sorted_logits[:, 0].unsqueeze(-1) - logits  # z_1 - z_j for all j
            return scores
        elif self.gamma == 2.0:  # Sparsemax case
            scores = torch.zeros(batch_size, num_classes, device=logits.device)
            for i in range(batch_size):
                for j in range(num_classes):
                    k = (indices[i] == j).nonzero(as_tuple=True)[0]
                    if k > 0:
                        scores[i, j] = torch.sum(sorted_logits[i, :k] - sorted_logits[i, k])
            return scores
        else:  # General gamma-entmax case
            
            scores = torch.zeros(batch_size, num_classes, device=logits.device)
            for i in range(batch_size):
                for j in range(num_classes):
                    k = (indices[i] == j).nonzero(as_tuple=True)[0]
                    if k > 0:
                        scores[i, j] = self._calculate_score_logsumexp(sorted_logits[i, :k] - sorted_logits[i, k])
                        
            return scores
        
    def _calculate_score_logsumexp(self, diffs):
        log_sum = torch.logsumexp(self.delta * torch.log(diffs), dim=0)
        return torch.exp(log_sum / self.delta)

