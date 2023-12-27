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
    return ((y_truth >= prediction_intervals[:, 0]) & (y_truth <= prediction_intervals[:, 1])).float().mean().item()


@METRICS_REGISTRY_REGRESSION.register()
def average_size(prediction_intervals):
    return torch.abs(prediction_intervals[:, 1] - prediction_intervals[:, 0]).mean().item()


class Metrics:
    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_REGRESSION.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_REGRESSION.get(metric)
