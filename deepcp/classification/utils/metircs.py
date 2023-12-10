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
def coverage_rate(prediction_sets, labels):
    cvg = 0
    for index, ele in enumerate(zip(prediction_sets, labels)):
        if ele[1] in ele[0]:
            cvg += 1
    return cvg / len(prediction_sets)


@METRICS_REGISTRY.register()
def average_size(prediction_sets, labels):
    avg_size = 0
    for index, ele in enumerate(prediction_sets):
        avg_size += len(ele)
    return avg_size / len(prediction_sets)

@METRICS_REGISTRY.register()
def CovGap(prediction_sets, labels,alpha,num_classes):
    rate_classes = np.zeros(num_classes)
    for k in range(num_classes):
        idx = np.where(labels == k)[0]
        selected_preds = [prediction_sets[i] for i in idx]
        rate_classes[k] = coverage_rate(selected_preds,labels[labels==k])
    
    return np.mean(np.abs(rate_classes-(1-alpha)))*100



class Metrics:
        
    def __call__(self, metric ) -> Any:
        if metric not in METRICS_REGISTRY.registered_names():
            raise NameError(f"The metric: {metric} is not defined in DeepCP.")
        return METRICS_REGISTRY.get(metric)
    
