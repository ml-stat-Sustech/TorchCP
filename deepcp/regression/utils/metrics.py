# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

from typing import Any
from deepcp.utils.registry import Registry

METRICS_REGISTRY = Registry("METRICS")


@METRICS_REGISTRY.register()
def coverage_rate(prediction_intervals, y_truth):
    y_truth = np.array(y_truth)
    return np.mean((y_truth >= prediction_intervals[:,0]) & (y_truth <= prediction_intervals[:,1]))


@METRICS_REGISTRY.register()
def average_size(prediction_intervals, y_truth):
    return np.mean(abs(prediction_intervals[:,1] - prediction_intervals[:,0]))




class Metrics:
        
    def __call__(self, metric ) -> Any:
        if metric not in METRICS_REGISTRY.registered_names():
            raise NameError(f"The metric: {metric} is not defined in DeepCP.")
        return METRICS_REGISTRY.get(metric)
    
