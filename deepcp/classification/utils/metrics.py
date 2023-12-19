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


#########################################
# Marginal coverage metric
#########################################

@METRICS_REGISTRY.register()
def coverage_rate(prediction_sets, labels):
    cvg = 0
    labels = labels.numpy()
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


#########################################
# Conditional coverage metric
#########################################

@METRICS_REGISTRY.register()
def CovGap(prediction_sets, labels,alpha,num_classes):
    rate_classes = []
    for k in range(num_classes):
        idx = np.where(labels == k)[0]
        selected_preds = [prediction_sets[i] for i in idx]
        if len(labels[labels==k]) !=0:
            rate_classes.append(coverage_rate(selected_preds,labels[labels==k]))
    rate_classes = np.array(rate_classes)
    return np.mean(np.abs(rate_classes-(1-alpha)))*100



@METRICS_REGISTRY.register()
def VioClasses(prediction_sets, labels,alpha,num_classes):
    violation_nums = 0
    for k in range(num_classes):
        if len(labels[labels==k]) ==0:
            violation_nums += 1
        else:
            idx = np.where(labels == k)[0]
            selected_preds = [prediction_sets[i] for i in idx]
            if coverage_rate(selected_preds,labels[labels==k]) < 1-alpha:
                violation_nums += 1
    return violation_nums


@METRICS_REGISTRY.register()
def DiffViolation(logits, prediction_sets, labels, alpha, num_classes):
    strata_diff = [[1,1],[2,3],[4,6],[7,10],[11,100],[101,1000]]
    correct_array  = np.zeros(len(labels))
    size_array  = np.zeros(len(labels))
    topk = []
    for index, ele in enumerate(logits):
        I = ele.argsort(descending = True)
        # print(labels[index])
        target = labels[index]
        topk.append(np.where((I - target.view(-1,1).numpy())==0)[1]+1) 
        correct_array[index] = 1 if labels[index] in prediction_sets[index] else 0
        size_array[index] = len(prediction_sets[index])
    topk  = np.concatenate(topk)
    
    ccss_diff = {}
    diff_violation =-1 
    
    for stratum in strata_diff:
        
        temp_index = np.argwhere( (topk >= stratum[0]) & (topk <= stratum[1]) )
        ccss_diff[str(stratum)]={}
        ccss_diff[str(stratum)]['cnt'] = len(temp_index)
        if len(temp_index) == 0:
            ccss_diff[str(stratum)]['cvg'] = 0
            ccss_diff[str(stratum)]['sz'] = 0
        else:
            temp_index= temp_index[:,0]
            cvg = np.round(np.mean(correct_array[temp_index]),3)
            sz  = np.round(np.mean(size_array[temp_index]),3)
            
            ccss_diff[str(stratum)]['cvg'] = cvg
            ccss_diff[str(stratum)]['sz'] = sz
            # 这个值越小越好
            stratum_violation = max(0,(1-alpha) -cvg)
            diff_violation = max(diff_violation, stratum_violation)
            
    diff_violation_one =0   
    for i in range(1,num_classes+1):
        temp_index = np.argwhere( topk == i )
        if len(temp_index)>0:
            temp_index= temp_index[:,0]
            # 这个值越小越好
            stratum_violation = max(0,(1-alpha) - np.mean(correct_array[temp_index]))
            diff_violation_one = max(diff_violation_one, stratum_violation)
    return diff_violation, diff_violation_one, ccss_diff


@METRICS_REGISTRY.register()
def SSCV(logits, prediction_sets, labels, alpha, stratified_size = [[0,1],[2,3],[4,10],[11,100],[101,1000]]):
    """Size-stratified coverage violation (SSCV)

    """
    size_array = np.zeros(len(labels))
    correct_array = np.zeros(len(labels))
    for index, ele in enumerate(prediction_sets):
        size_array[index] = len(ele)
        correct_array[index] = 1 if labels[index] in prediction_sets[index] else 0
        
    sscv = -1 
    for stratum in stratified_size:
        temp_index = np.argwhere( (size_array >= stratum[0]) & (size_array <= stratum[1]) )
        if len(temp_index) > 0:
           
            stratum_violation = max(0,(1-alpha) - np.mean(correct_array[temp_index]))
            sscv = max(sscv, stratum_violation)
    return sscv
                
                
                
class Metrics:
        
    def __call__(self, metric ) -> Any:
        if metric not in METRICS_REGISTRY.registered_names():
            raise NameError(f"The metric: {metric} is not defined in DeepCP.")
        return METRICS_REGISTRY.get(metric)
    
