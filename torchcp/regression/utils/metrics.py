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
    num_columns = prediction_intervals.shape[1]
    assert num_columns % 2 == 0, f"The number of columns in prediction_intervals must be even, but got {num_columns}"

    condition = torch.zeros_like(y_truth, dtype=torch.bool)

    for i in range(num_columns // 2):
        lower_bound = prediction_intervals[:, 2*i]
        upper_bound = prediction_intervals[:, 2*i+1]
        condition |= torch.bitwise_and(y_truth >= lower_bound, y_truth <= upper_bound)

    coverage = torch.sum(condition).item()
    return coverage / len(y_truth)


@METRICS_REGISTRY_REGRESSION.register()
def average_size(prediction_intervals):
    num_columns = prediction_intervals.shape[1]
    assert num_columns % 2 == 0, f"The number of columns in prediction_intervals must be even, but got {num_columns}"

    size = torch.abs(prediction_intervals[:, 1::2] - prediction_intervals[:, 0::2]).sum(dim=1)
    average_size = size.mean().item()

    return average_size


class Metrics:
    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_REGRESSION.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_REGRESSION.get(metric)
