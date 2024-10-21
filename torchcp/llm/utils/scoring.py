"""Implementation of different scoring variants."""

import torch


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
    shifted = torch.pad(p, ((0, 0), (1, 0)), constant_values=1)[:, :-1]
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
