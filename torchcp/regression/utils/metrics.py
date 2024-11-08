# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from typing import Any

from torchcp.utils.registry import Registry

METRICS_REGISTRY_REGRESSION = Registry("METRICS")


@METRICS_REGISTRY_REGRESSION.register()
def coverage_rate(prediction_intervals, y_truth):
    """
    Calculates the coverage rate of prediction intervals.
    
    Args:
        prediction_intervals (torch.Tensor): A tensor of shape (batch_size, num_intervals * 2),
                                             where each interval has a lower and upper bound.
        y_truth (torch.Tensor): A tensor of ground truth values of shape (batch_size,).
    
    Returns:
        torch.Tensor: The coverage rate, representing the proportion of ground truth values
                      within the specified prediction intervals.
    
    Raises:
        AssertionError: If the number of columns in prediction_intervals is not even.
    """
    num_columns = prediction_intervals.shape[-1]
    assert num_columns % 2 == 0, f"The number of columns in prediction_intervals must be even, but got {num_columns}"

    if len(prediction_intervals.shape) == 2:
        prediction_intervals = prediction_intervals.unsqueeze(1)
    if len(y_truth.shape) == 1:
        y_truth = y_truth.unsqueeze(1)

    condition = torch.zeros_like(y_truth, dtype=torch.bool)

    for i in range(num_columns // 2):
        lower_bound = prediction_intervals[..., 2 * i]
        upper_bound = prediction_intervals[..., 2 * i + 1]
        condition |= torch.bitwise_and(y_truth >= lower_bound, y_truth <= upper_bound)

    coverage_rate = torch.sum(condition, dim=0).cpu() / y_truth.shape[0]
    return coverage_rate


@METRICS_REGISTRY_REGRESSION.register()
def average_size(prediction_intervals):
    """
    Computes the average size of prediction intervals.
    
    Args:
        prediction_intervals (torch.Tensor): A tensor of shape (batch_size, num_intervals * 2),
                                             where each interval has a lower and upper bound.
    
    Returns:
        torch.Tensor: The average size of the prediction intervals across all samples.
    
    Raises:
        AssertionError: If the number of columns in prediction_intervals is not even.
    """
    num_columns = prediction_intervals.shape[-1]
    assert num_columns % 2 == 0, f"The number of columns in prediction_intervals must be even, but got {num_columns}"

    size = torch.abs(prediction_intervals[..., 1::2] - prediction_intervals[..., 0::2]).sum(dim=-1)
    average_size = size.mean(dim=0).cpu()

    return average_size


class Metrics:
    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_REGRESSION.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_REGRESSION.get(metric)
