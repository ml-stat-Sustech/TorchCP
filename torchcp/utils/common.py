# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import math
import numpy as np
import random
import torch
import warnings

__all__ = ["calculate_conformal_value", "get_device", "DimensionError"]

class DimensionError(Exception):
    pass


def get_device(model):
    """
    Get the device of a PyTorch model.

    This function determines the device (CPU or GPU) on which the model's parameters are located.
    If the model is None, it defaults to using GPU if available, otherwise it uses CPU.

    Args:
        model (torch.nn.Module or None): A PyTorch model. If None, the function checks for GPU availability.

    Returns:
        torch.device: The device on which the model's parameters are located, or the default device (CPU or GPU).
    """
    if model is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = next(model.parameters()).device
    return device


def calculate_conformal_value(scores, alpha, default_q_hat=torch.inf):
    """
    Calculate the 1-alpha quantile of scores for conformal prediction.

    This function computes the threshold value (quantile) used to construct prediction sets based on the given
    non-conformity scores and significance level alpha. If the scores are empty or the quantile value exceeds 1,
    it returns the default_q_hat value.

    Args:
        scores (torch.Tensor): Non-conformity scores.
        alpha (float): Significance level, must be between 0 and 1.
        default_q_hat (torch.Tensor or str, optional): Default threshold value to use if scores are empty or invalid.
            If set to "max", it uses the maximum value of scores. Default is torch.inf.

    Returns:
        torch.Tensor: The threshold value used to construct prediction sets.
    
    Raises:
        ValueError: If alpha is not between 0 and 1.
    """
    if default_q_hat == "max":
        default_q_hat = torch.max(scores)
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Significance level 'alpha' must be in [0,1].")
    if len(scores) == 0:
        warnings.warn(
            f"The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    N = scores.shape[0]
    qunatile_value = math.ceil((N + 1) * (1 - alpha)) / N
    if qunatile_value > 1:
        warnings.warn(
            f"The value of quantile exceeds 1. It should be a value in [0,1]. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat

    return torch.quantile(scores, qunatile_value, dim=0, interpolation='lower').to(scores.device)
