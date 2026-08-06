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