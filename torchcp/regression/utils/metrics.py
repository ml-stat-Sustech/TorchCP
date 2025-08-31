# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any

import torch

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
    
    """
    num_columns = prediction_intervals.shape[-1]
    if num_columns % 2 != 0:
        raise ValueError(f"The number of columns in prediction_intervals must be even, but got {num_columns}")

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
    return coverage_rate.item()


@METRICS_REGISTRY_REGRESSION.register()
def average_size(prediction_intervals):
    """
    Computes the average size of prediction intervals.
    
    Args:
        prediction_intervals (torch.Tensor): A tensor of shape (batch_size, num_intervals * 2),
                                             where each interval has a lower and upper bound.
    
    Returns:
        torch.Tensor: The average size of the prediction intervals across all samples.
    """
    num_columns = prediction_intervals.shape[-1]
    if num_columns % 2 != 0:
        raise ValueError(f"The number of columns in prediction_intervals must be even, but got {num_columns}")

    size = torch.abs(prediction_intervals[..., 1::2] - prediction_intervals[..., 0::2]).sum(dim=-1)
    average_size = size.mean(dim=0).cpu().item()

    return average_size


@METRICS_REGISTRY_REGRESSION.register()
def false_discovery_proportion(y_truth, thresholds, indices):
    """
    Conpute the false discovery proportion (the proportion of false discovery among all selected points) of the
    selection set.

    Args:
        y_truth (torch.Tensor): A tensor of ground truth values.
        thresholds (torch.Tensor): Tensor of user-defined thresholds.
        indices (torch.Tensor): A tensor containing the indices of selected points.

    Returns:
        torch.Tensor: The false discovery proportion of the selection set.
    """
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)

    false_positives = torch.sum(y_truth[indices] <= thresholds[indices])
    fdp = false_positives / indices.shape[-1] if indices.shape[-1] > 0 else torch.tensor(0.)
    return fdp.item()


@METRICS_REGISTRY_REGRESSION.register()
def power(y_truth, thresholds, indices):
    """
        Conpute the power (the proportion of desirable points that are correctly selected) of the selection set.

        Args:
            y_truth (torch.Tensor): A tensor of ground truth values.
            thresholds (torch.Tensor): Tensor of user-defined thresholds.
            indices (torch.Tensor): A tensor containing the indices of selected points.

        Returns:
            torch.Tensor: The power of the selection set.
        """
    if indices.dim() == 0:
        indices = indices.unsqueeze(0)

    true_positives = torch.sum(y_truth[indices] > thresholds[indices])
    power = true_positives / torch.sum(y_truth > thresholds)
    return power.item()


class Metrics:
    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_REGRESSION.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_REGRESSION.get(metric)
