# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from typing import Any

from torchcp.utils.registry import Registry

METRICS_REGISTRY_LLM = Registry("METRICS")


@METRICS_REGISTRY_LLM.register()
def average_size(prediction_sets):
    assert len(prediction_sets) > 0, "The number of prediction set must be greater than 0."

    size_avg = torch.mean(prediction_sets)
    return size_avg 


@METRICS_REGISTRY_LLM.register()
def average_sample_size(prediction_sets):
    """the average number of sample size"""
    assert len(prediction_sets) > 0, "The number of prediction set must be greater than 0."

    max_indices = prediction_sets.shape[1] - 1 - torch.argmax(prediction_sets.flip(1), dim=1)
    return torch.mean(max_indices.float()) 


@METRICS_REGISTRY_LLM.register()
def average_set_loss(prediction_sets, prediction_set_loss):
    """the average number of sample size"""
    assert len(prediction_sets) > 0, "The number of prediction set must be greater than 0."
    
    max_indices = prediction_sets.shape[1] - 1 - torch.argmax(prediction_sets.flip(1).to(torch.int), dim=1)

    losses =  prediction_set_loss[torch.arange(prediction_sets.shape[0]), max_indices]
    return torch.mean(losses)


@METRICS_REGISTRY_LLM.register()
def SSCL(prediction_sets, prediction_set_loss, num_bins=20):
    """
    Size-stratified conditional loss.
    
    Paper: Conformal Language Modeling (Victor Quach et al., ICLR'24)
    """
    
    prediction_sets = torch.tensor(prediction_set_loss, dtype=prediction_set_loss.dtype)

    prediction_sizes = average_size(prediction_sets)
    bins = torch.quantile(prediction_sizes, torch.linspace(0, 1, num_bins, dtype= prediction_sets.dtype))
    binids = torch.bucketize(prediction_sizes, torch.cat([torch.tensor([0]), torch.unique(bins)]))

    L_worst_avg = -1
    for binid in torch.unique(binids):
        kept = binids == binid
        num_kept_examples = torch.maximum(torch.sum(kept), torch.tensor(1, device=kept.device))
        Ls_mask_avg = torch.sum(prediction_set_loss * kept) / num_kept_examples
        L_worst_avg = max(L_worst_avg, Ls_mask_avg)

class Metrics:

    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_LLM.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_LLM.get(metric)
