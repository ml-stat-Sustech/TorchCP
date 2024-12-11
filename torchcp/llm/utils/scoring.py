# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Implementation of different scoring variants."""

import torch

import torch.nn.functional as F


def geometric(p, mask=None):
    """Score of a set is based on a geometric distribution approximation:

    p(\exists y \in C : L(y) = 0) = 1 - \prod 1 - p(A(y_i) = 1)

    This is consistent with -sum \log (1 - p(A(y_i) = 1)).

    Args:
        p: Matrix of size [num_examples, max_size].
           Each entry of p approximates p(A(y_ij) = 1).

    Returns:
        Log geometric scores.
    """
    if mask is not None:
        p = p * mask
    p = torch.maximum(1 - p, torch.tensor(1e-8))
    return -torch.cumsum(torch.log(p), dim=-1)


def marginal(p, mask=None):
    """Similar to geometric, but with p(y_k is the only y with A(y) = 1)."""
    if mask is not None:
        p = p * mask
    p = torch.maximum(1 - p, torch.tensor(1e-8))
    shifted = F.pad(p, (1, 0), mode='constant', value=1.0)[:, :-1]
    return -torch.log(1 - p) - torch.cumsum(torch.log(shifted), dim=-1)


def first_k(X, mask=None):
    """Scores are equal to the number of draws."""
    num_examples, max_generations = X.shape
    scores = torch.ones((num_examples, max_generations))
    if mask is not None:
        scores = scores * mask
    return torch.cumsum(scores, dim=-1)


def first_k_no_mask(X, mask=None):
    """Scores are equal to the number of draws."""
    del mask
    num_examples, max_generations = X.shape
    scores = torch.ones((num_examples, max_generations))
    return torch.cumsum(scores, axis=-1)


def max(X, mask=None):
    if mask is not None:
        X = X * mask
    cumulative_max, _ = torch.cummax(X, dim=-1)
    return cumulative_max


def sum(X, mask=None):
    if mask is not None:
        X = X * mask
    return torch.cumsum(X, dim=-1)
