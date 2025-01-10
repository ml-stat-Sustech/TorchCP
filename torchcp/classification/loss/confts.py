# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__all__ = ["ConfTS"]

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from torchcp.classification.loss.base import BaseLoss


class ConfTS(BaseLoss):
    """
    Conformal Temperature Scaling (ConfTS).
        
    The class implements the loss function of conformal temperature scaling. It supports
    multiple loss functions and allows for flexible configuration of the training
    process.

    Args:
        predictor (torchcp.classification.Predictor): An instance of the CP predictor class.
        fraction (float): The fraction of the calibration set in each training batch.
            Must be a value in (0, 1). Default is 0.5.
        soft_qunatile (bool, optional): Whether to use soft quantile. Default is True.

    Examples::
        >>> predictor = torchcp.classification.SplitPredictor(score_function=APS(score_type="softmax", randomized=False))
        >>> confts = ConfTS(predictor=predictor, fraction=0.2)
        >>> logits = torch.randn(100, 10)
        >>> labels = torch.randint(0, 2, (100,))
        >>> loss = confts(logits, labels)
        >>> loss.backward()
        
    Reference:
        Xi et al. "Delving into Temperature Scaling for Adaptive Conformal Prediction" (2023), https://arxiv.org/abs/2402.04344
        
    """

    def __init__(self, predictor, alpha, fraction=0.5, soft_qunatile=True):

        super(ConfTS, self).__init__(predictor)

        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0,1).")

        if not (0 < fraction < 1):
            raise ValueError("fraction should be a value in (0,1).")

        self.predictor = predictor
        self.soft_qunatile = soft_qunatile
        self.fraction = fraction
        self.alpha = alpha
        self.device = predictor.get_device()

    def forward(self, logits, labels):
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        # Compute Size Loss
        val_split = int(self.fraction * logits.shape[0])
        cal_logits = logits[:val_split]
        cal_labels = labels[:val_split]
        test_logits = logits[val_split:]
        test_labels = labels[val_split:]

        if self.soft_qunatile:
            cal_scores = self.predictor.score_function(cal_logits, cal_labels)
            tau = self._soft_quantile(cal_scores, self.alpha)
        else:
            self.predictor.calculate_threshold(cal_logits.detach(), cal_labels.detach(), self.alpha)
            tau = self.predictor.q_hat

        test_scores = self.predictor.score_function(test_logits)

        return self.compute_loss(test_scores, test_labels, tau)

    def compute_loss(self, test_scores, test_labels, tau):
        return torch.mean((tau - test_scores[range(test_scores.shape[0]), test_labels]) ** 2)

    def __neural_sort(self,
                      scores: Tensor,
                      tau: float = 0.1,
                      ) -> Tensor:
        """
        Soft sorts scores (descending) along last dimension
        Follows implementation form
        https://github.com/ermongroup/neuralsort/blob/master/pytorch/neuralsort.py
        
        Grover, Wang et al., Stochastic Optimization of Sorting Networks via Continuous Relaxations

        Args:
            scores (Tensor): scores to sort
            tau (float, optional): smoothness factor. Defaults to 0.01.
        Returns:
            Tensor: permutation matrix such that sorted_scores = P @ scores 
        """
        pairwise_abs_diffs = (scores[..., :, None] - scores[..., None, :]).abs()
        n = scores.shape[-1]

        pairwise_abs_diffs_sum = pairwise_abs_diffs @ torch.ones(n, 1, device=pairwise_abs_diffs.device)
        scores_diffs = scores[..., :, None] * (
                n - 1 - 2 * torch.arange(n, device=pairwise_abs_diffs.device, dtype=torch.float))
        P_scores = (scores_diffs - pairwise_abs_diffs_sum).transpose(-2, -1)
        P_hat = torch.softmax(P_scores / tau, dim=-1)

        return P_hat

    def _soft_quantile(self, scores: Tensor,
                       q: float,
                       dim=-1,
                       **kwargs
                       ) -> Tensor:
        # swap requested dim with final dim
        dims = list(range(len(scores.shape)))
        dims[-1], dims[dim] = dims[dim], dims[-1]
        scores = scores.permute(*dims)
        # normalize scores on last dimension
        # scores_norm = (scores - scores.mean()) / 3.*scores.std()
        # obtain permutation matrix for scores
        P_hat = self.__neural_sort(scores, **kwargs)
        # use permutation matrix to sort scores
        sorted_scores = (P_hat @ scores[..., None])[..., 0]
        # turn quantiles into indices to select
        n = scores.shape[-1]
        squeeze = False
        if isinstance(q, float):
            squeeze = True
            q = [q]
        q = torch.tensor(q, dtype=torch.float, device=scores.device)

        indices = (1 - q) * (n + 1) - 1
        indices_low = torch.floor(indices).long()
        indices_frac = indices - indices_low
        indices_high = indices_low + 1
        # select quantiles from computed scores:
        quantiles = sorted_scores[..., torch.cat([indices_low, indices_high])]
        quantiles = quantiles[..., :q.shape[0]] + indices_frac * (
                quantiles[..., q.shape[0]:] - quantiles[..., :q.shape[0]])
        # restore dimension order
        if len(dims) > 1:
            quantiles = quantiles.permute(*dims)

        if squeeze:
            quantiles = quantiles.squeeze(dim)

        return quantiles
