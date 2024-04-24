# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Any

import numpy as np

from torchcp.utils.registry import Registry

METRICS_REGISTRY_CLASSIFICATION = Registry("METRICS")


#########################################
# Marginal coverage metric
#########################################

@METRICS_REGISTRY_CLASSIFICATION.register()
def coverage_rate(prediction_sets, labels, coverage_type="default", num_classes = None):
    """The metric for coverage.

    Args:
        prediction_sets (List): _description_
        labels (list): _description_
        coverage_type (str, optional): the type of coverage rate. Defaults to "default". Options are 'default' (the marginal coverage rate), 'macro' (the average coverage rate of all classes).
        num_classes (_type_, optional): the number of classes. When coverage_type == 'macro", you must define the number of classes.

    Returns:
        float: the empirical coverage rate.
    """
    assert len(prediction_sets)>0, "The number of prediction set must be greater than 0."
    labels = labels.cpu()
    cvg = 0
    
    if coverage_type == "macro":
        assert (num_classes!=None), "Macro Coverage metric needs the number of classes."
        rate_classes = []
        for k in range(num_classes):
            idx = np.where(labels == k)[0]
            selected_preds = [prediction_sets[i] for i in idx]
            if len(labels[labels == k]) != 0:
                rate_classes.append(coverage_rate(selected_preds, labels[labels == k]))
            else:
                # If there is no the "k" class in the "labels", we skip the calculation of this class.
                continue
        cvg = np.mean(rate_classes)
    else:
        for index, ele in enumerate(zip(prediction_sets, labels)):
            if ele[1] in ele[0]:
                cvg += 1
        cvg = cvg / len(prediction_sets)
    return cvg


@METRICS_REGISTRY_CLASSIFICATION.register()
def average_size(prediction_sets, labels):
    assert len(prediction_sets)>0, "The number of prediction set must be greater than 0."

    labels = labels.cpu()
    avg_size = 0
    for index, ele in enumerate(prediction_sets):
        avg_size += len(ele)
    return avg_size / len(prediction_sets)


#########################################
# Conditional coverage metric
#########################################

@METRICS_REGISTRY_CLASSIFICATION.register()
def CovGap(prediction_sets, labels, alpha, num_classes, shot_idx = None):
    assert len(prediction_sets)>0, "The number of prediction set must be greater than 0."
    labels = labels.cpu()
    cls_coverage = []
    for k in range(num_classes):
        idx = np.where(labels == k)[0]
        selected_preds = [prediction_sets[i] for i in idx]
        
        if len(labels[labels == k]) != 0:
            the_coverage = coverage_rate(selected_preds, labels[labels == k])
        else:
            the_coverage = 0
        cls_coverage.append(the_coverage)
            
    cls_coverage = np.array(cls_coverage)
    overall_covgap = np.mean(np.abs(cls_coverage - (1 - alpha))) * 100
    
    if shot_idx == None:
        return overall_covgap
    covgaps = [overall_covgap]
    for shot in shot_idx:
        shot_covgap =  np.mean(np.abs(cls_coverage[shot] - (1 - alpha))) * 100
        covgaps.append(shot_covgap)
       
    return covgaps


@METRICS_REGISTRY_CLASSIFICATION.register()
def VioClasses(prediction_sets, labels, alpha, num_classes):
    labels = labels.cpu()
    violation_nums = 0
    for k in range(num_classes):
        if len(labels[labels == k]) == 0:
            violation_nums += 1
        else:
            idx = np.where(labels == k)[0]
            selected_preds = [prediction_sets[i] for i in idx]
            if coverage_rate(selected_preds, labels[labels == k]) < 1 - alpha:
                violation_nums += 1
    return violation_nums


@METRICS_REGISTRY_CLASSIFICATION.register()
def DiffViolation(logits, prediction_sets, labels, alpha, num_classes):
    """
    Difficulty-stratified coverage violation
    """
    labels = labels.cpu()
    strata_diff = [[1, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]]
    correct_array = np.zeros(len(labels))
    size_array = np.zeros(len(labels))
    topk = []
    for index, ele in enumerate(logits):
        I = ele.argsort(descending=True)
        target = labels[index]
        topk.append(np.where((I - target.view(-1, 1).numpy()) == 0)[1] + 1)
        correct_array[index] = 1 if labels[index] in prediction_sets[index] else 0
        size_array[index] = len(prediction_sets[index])
    topk = np.concatenate(topk)

    ccss_diff = {}
    diff_violation = -1

    for stratum in strata_diff:

        temp_index = np.argwhere((topk >= stratum[0]) & (topk <= stratum[1]))
        ccss_diff[str(stratum)] = {}
        ccss_diff[str(stratum)]['cnt'] = len(temp_index)
        if len(temp_index) == 0:
            ccss_diff[str(stratum)]['cvg'] = 0
            ccss_diff[str(stratum)]['sz'] = 0
        else:
            temp_index = temp_index[:, 0]
            cvg = np.round(np.mean(correct_array[temp_index]), 3)
            sz = np.round(np.mean(size_array[temp_index]), 3)

            ccss_diff[str(stratum)]['cvg'] = cvg
            ccss_diff[str(stratum)]['sz'] = sz
            stratum_violation = max(0, (1 - alpha) - cvg)
            diff_violation = max(diff_violation, stratum_violation)

    diff_violation_one = 0
    for i in range(1, num_classes + 1):
        temp_index = np.argwhere(topk == i)
        if len(temp_index) > 0:
            temp_index = temp_index[:, 0]
            stratum_violation = max(0, (1 - alpha) - np.mean(correct_array[temp_index]))
            diff_violation_one = max(diff_violation_one, stratum_violation)
    return diff_violation, diff_violation_one, ccss_diff


@METRICS_REGISTRY_CLASSIFICATION.register()
def SSCV(prediction_sets, labels, alpha, stratified_size=[[0, 1], [2, 3], [4, 10], [11, 100], [101, 1000]]):
    """
    Size-stratified coverage violation (SSCV)
    """
    labels = labels.cpu()
    size_array = np.zeros(len(labels))
    correct_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        size_array[index] = len(ele)
        correct_array[index] = 1 if labels[index] in ele else 0

    sscv = -1
    for stratum in stratified_size:
        temp_index = np.argwhere((size_array >= stratum[0]) & (size_array <= stratum[1]))
        if len(temp_index) > 0:
            stratum_violation = abs((1 - alpha) - np.mean(correct_array[temp_index]))
            sscv = max(sscv, stratum_violation)
    return sscv


class Metrics:

    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_CLASSIFICATION.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_CLASSIFICATION.get(metric)
