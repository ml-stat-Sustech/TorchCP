# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from typing import Any

from torchcp.utils.registry import Registry

from torchcp.classification.utils.metrics import coverage_rate as graph_coverage_rate
from torchcp.classification.utils.metrics import average_size as graph_average_size


METRICS_REGISTRY_GRAPH = Registry("METRICS")

#########################################
# Marginal coverage metric
#########################################


@METRICS_REGISTRY_GRAPH.register()
def coverage_rate(prediction_sets, labels, coverage_type="default", num_classes=None):
    return graph_coverage_rate(prediction_sets, labels, coverage_type=coverage_type, num_classes=num_classes)



@METRICS_REGISTRY_GRAPH.register()
def average_size(prediction_sets, labels):
    return graph_average_size(prediction_sets, labels)


@METRICS_REGISTRY_GRAPH.register()
def singleton_hit_ratio(prediction_sets, labels):
    assert len(
        prediction_sets) > 0, "The number of prediction set must be greater than 0."
    n = len(prediction_sets)

    one_size = 0
    for index, ele in enumerate(prediction_sets):
        if len(ele) == 1 and ele[0] == labels[index]:
            one_size += 1
    return one_size / n


class Metrics:

    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_GRAPH.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_GRAPH.get(metric)
