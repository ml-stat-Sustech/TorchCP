# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torchcp.utils.registry import Registry

METRICS_REGISTRY_CLASSIFICATION = Registry("METRICS")


#########################################
# Marginal coverage metric
#########################################

@METRICS_REGISTRY_CLASSIFICATION.register()
def coverage_rate(prediction_sets, labels, coverage_type="default", num_classes=None):
    """
    The metric for empirical coverage.
    
    Args:
        prediction_sets (torch.Tensor): Boolean tensor of prediction sets (N x C), where N is number of samples
            and C is number of classes.
        labels (torch.Tensor): Ground-truth labels (N,).
        coverage_type (str, optional): Type of coverage rate calculation.
            'default': marginal coverage rate
            'macro': average coverage rate across all classes
        num_classes (int, optional): Number of classes. Required when coverage_type is 'macro'.
    Returns:
        float: Empirical coverage rate.
    """
    labels = labels.cpu()
    prediction_sets = prediction_sets.cpu()

    if prediction_sets.shape[0] != len(labels):
        raise ValueError("The number of prediction sets must be equal to the number of labels.")

    if coverage_type not in ["default", "macro"]:
        raise ValueError("coverage_type must be 'default' or 'macro'.")

    cvg = 0

    covered = prediction_sets[torch.arange(len(labels)), labels]

    if coverage_type == "macro":
        if num_classes is None:
            raise ValueError("When coverage_type is 'macro', you must define the number of classes.")

        class_counts = torch.bincount(labels, minlength=num_classes)
        class_covered = torch.bincount(labels[covered == 1], minlength=num_classes)

        class_coverage = torch.zeros_like(class_counts, dtype=torch.float)
        valid_classes = class_counts > 0
        class_coverage[valid_classes] = class_covered[valid_classes].float() / class_counts[valid_classes].float()

        cvg = class_coverage.mean().item()
    else:
        cvg = covered.float().mean().item()
    return cvg


@METRICS_REGISTRY_CLASSIFICATION.register()
def average_size(prediction_sets, labels=None):
    return torch.mean(torch.sum(prediction_sets, dim=1).float()).item()


#########################################
# Conditional coverage metric
#########################################

@METRICS_REGISTRY_CLASSIFICATION.register()
def CovGap(prediction_sets, labels, alpha, num_classes, shot_idx=None):
    """
    The average class-conditional coverage gap.

    Paper: Class-Conditional Conformal Prediction with Many Classes (Ding et al., 2023)
    Link: https://neurips.cc/virtual/2023/poster/70548
    
    Args:
        prediction_sets (torch.Tensor): Boolean tensor of prediction sets (N x C).
        labels (torch.Tensor): Ground-truth labels (N,).
        alpha (float): User-guided confidence level.
        num_classes (int): Number of classes.
        shot_idx (list, optional): Indices of classes to compute coverage gap.
    
    Returns:
        float: Average class-conditional coverage gap (percentage).
    """
    if prediction_sets.shape[0] != len(labels):
        raise ValueError("Number of prediction sets must match number of labels")

    labels = labels.cpu()
    prediction_sets = prediction_sets.cpu()

    covered = prediction_sets[torch.arange(len(labels)), labels]
    class_counts = torch.bincount(labels, minlength=num_classes)
    class_covered = torch.bincount(labels[covered == 1], minlength=num_classes)

    cls_coverage_rate = torch.zeros_like(class_counts, dtype=torch.float32)
    valid_classes = class_counts > 0
    cls_coverage_rate[valid_classes] = class_covered[valid_classes].float() / class_counts[valid_classes].float()

    if shot_idx is not None:
        mask = torch.zeros(num_classes, dtype=torch.bool)
        mask[shot_idx] = True
        valid_classes = valid_classes & mask
        
    overall_covgap = torch.mean(torch.abs(cls_coverage_rate[valid_classes] - (1 - alpha))) * 100
    return overall_covgap.float().item()


@METRICS_REGISTRY_CLASSIFICATION.register()
def VioClasses(prediction_sets, labels, alpha, num_classes):
    """
    The number of violated classes.

    Paper: Empirically Validating Conformal Prediction on Modern Vision Architectures Under Distribution Shift and Long-tailed Data (Kasa et al., 2023)
    Link: https://arxiv.org/abs/2307.01088
    
    Args:
        prediction_sets (torch.Tensor): Boolean tensor of prediction sets (N x C).
        labels (torch.Tensor): Ground-truth labels (N,).
        alpha (float): User-guided confidence level.
        num_classes (int): Number of classes.
    
    Returns:
        int: Number of classes with violated coverage.
    """
    if prediction_sets.shape[0] != len(labels):
        raise ValueError("Number of prediction sets must match number of labels")

    labels = labels.cpu()
    prediction_sets = prediction_sets.cpu()
    violation_nums = 0

    covered = prediction_sets[torch.arange(len(labels)), labels]

    class_covered = torch.bincount(labels[covered == 1], minlength=num_classes)
    class_counts = torch.bincount(labels, minlength=num_classes)

    valid_classes = class_counts > 0
    class_coverage = torch.zeros(num_classes, dtype=torch.float32)
    class_coverage[valid_classes] = class_covered[valid_classes].float() / class_counts[valid_classes].float()
    
    violation_nums = torch.sum(class_coverage[valid_classes] < (1 - alpha))
    return violation_nums.item()


@METRICS_REGISTRY_CLASSIFICATION.register()
def DiffViolation(logits, prediction_sets, labels, alpha,
                  strata_diff=[[1, 1], [2, 3], [4, 6], [7, 10], [11, 100], [101, 1000]]):
    """
    Difficulty-stratified coverage violation

    Paper: Uncertainty Sets for Image Classifiers using Conformal Prediction (Angelopoulos et al., 2020)
    Link: https://arxiv.org/abs/2009.14193
    
    
    Args:
        logits (torch.Tensor): the predicted logits.
        prediction_sets (torch.Tensor): the prediction sets generated by CP algorithms.
        labels (list): the ground-truth label of each samples.
        alpha (float): the user-guided confidence level.
        strata_diff (list): a coarse partitioning of the possible difficulties.

    Returns:
        2-tuple: (the difficulty-stratified coverage violation, the number of samples, the empirical coverage and size of each difficulty).
    """
    if prediction_sets.shape[0] != len(labels):
        raise ValueError("Number of prediction sets must match number of labels")
    if not isinstance(strata_diff, list):
        raise TypeError("strata_diff must be a list")

    labels = labels.cpu()
    logits = logits.cpu()
    prediction_sets = prediction_sets.cpu()

    covered = prediction_sets[torch.arange(len(labels)), labels]
    set_sizes = prediction_sets.sum(dim=1)

    sorted_indices = torch.argsort(logits, dim=1, descending=True)
    topk = torch.zeros(len(labels), dtype=torch.int32)
    for i, (indices, label) in enumerate(zip(sorted_indices, labels)):
        topk[i] = (indices == label).nonzero(as_tuple=True)[0].item() + 1

    ccss_diff = {}
    diff_violation = -1

    for stratum in strata_diff:
        stratum_mask = (topk >= stratum[0]) & (topk <= stratum[1])
        stratum_size = stratum_mask.sum().item()

        ccss_diff[str(stratum)] = {'cnt': stratum_size}

        if stratum_size == 0:
            ccss_diff[str(stratum)].update({'cvg': 0, 'sz': 0})
            continue

        cvg = covered[stratum_mask].float().mean().item()
        sz = set_sizes[stratum_mask].float().mean().item()

        ccss_diff[str(stratum)].update({
            'cvg': round(cvg, 3),
            'sz': round(sz, 3)
        })
        stratum_violation = abs(1 - alpha - cvg)
        diff_violation = max(diff_violation, stratum_violation)

    return diff_violation, ccss_diff


@METRICS_REGISTRY_CLASSIFICATION.register()
def SSCV(prediction_sets, labels, alpha, stratified_size=[[0, 1], [2, 3], [4, 10], [11, 100], [101, 1000]]):
    """
    Size-stratified coverage violation (SSCV).
    
    Paper: Uncertainty Sets for Image Classifiers using Conformal Prediction (Angelopoulos et al., 2020)
    
    Link : https://iclr.cc/virtual/2021/spotlight/3435
    
    Args:
        prediction_sets (torch.Tensor): Boolean tensor of prediction sets (N x C).
        labels (torch.Tensor): Ground-truth labels (N,).
        alpha (float): User-guided confidence level (between 0 and 1).
        stratified_size (list): Coarse partitioning of possible set sizes.
            Each element should be a list [min_size, max_size] where:
            - min_size and max_size are non-negative integers
            - min_size <= max_size
            - Ranges should not overlap
    
    Returns:
        Int: The value of SSCV.
    
    """
    if len(prediction_sets) != len(labels):
        raise ValueError("The number of prediction sets must be equal to the number of labels.")

    if not isinstance(stratified_size, list) or not stratified_size:
        raise ValueError("stratified_size must be a non-empty list")

    labels = labels.cpu()
    prediction_sets = prediction_sets.cpu()

    size_array = prediction_sets.sum(dim=1)  # shape: (N,)
    correct_array = prediction_sets[torch.arange(len(labels)), labels]  # shape: (N,)

    sscv = -1
    for stratum in stratified_size:
        stratum_mask = (size_array >= stratum[0]) & (size_array <= stratum[1])
        stratum_size = stratum_mask.sum()

        if stratum_size > 0:
            stratum_coverage = correct_array[stratum_mask].float().mean()
            stratum_violation = torch.abs(1 - alpha - stratum_coverage).item()
            sscv = max(sscv, stratum_violation)
    return sscv


@METRICS_REGISTRY_CLASSIFICATION.register()
def WSC(features, prediction_sets, labels, delta=0.1, M=1000, test_fraction=0.75, random_state=2020, verbose=False):
    """
    Worst-Slice Coverage (WSC).
    
     Classification with Valid and Adaptive Coverage (Romano et al., 2020)
     Paper: Classification with Valid and Adaptive Coverage
     Link : https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html
     Code: https://github.com/msesia/arc/tree/d80d27519f18b11e7feaf8cf0da8827151af9ce3

    
    Args:
        features (torch.Tensor): Input features (N x D).
        prediction_sets (torch.Tensor): Boolean tensor of prediction sets (N x C).
        y (torch.Tensor): Ground-truth labels (N,).
        delta (float): Confidence level (between 0 and 1).
        M (int): Number of random projections.
        test_size (float): Proportion of test split.
        random_state (int): Random seed.
        verbose (bool): Whether to print progress.
    
     Returns:
         Float: the value of unbiased WSV.
    
    """

    if not 0 < delta < 1:
        raise ValueError("delta must be between 0 and 1")
    if not 0 < test_fraction < 1:
        raise ValueError("test_size must be between 0 and 1")
    if M <= 0:
        raise ValueError("M must be positive")

    if len(features.shape) != 2:
        raise ValueError(f"features must be 2D tensor, got shape {features.shape}")
    if len(prediction_sets.shape) != 2:
        raise ValueError(f"prediction_sets must be 2D tensor, got shape {prediction_sets.shape}")
    if len(labels.shape) != 1:
        raise ValueError(f"labels must be 1D tensor, got shape {labels.shape}")

    if features.shape[0] != len(labels):
        raise ValueError(
            f"Number of samples mismatch: features has {features.shape[0]} samples but labels has {len(labels)} samples")
    if features.shape[0] != prediction_sets.shape[0]:
        raise ValueError(
            f"Number of samples mismatch: features has {features.shape[0]} samples but prediction_sets has {prediction_sets.shape[0]} samples")
    if prediction_sets.shape[1] != len(torch.unique(labels)):
        raise ValueError(
            f"Number of classes mismatch: prediction_sets has {prediction_sets.shape[1]} classes but labels has {len(torch.unique(labels))} unique classes")

    features = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    covered = prediction_sets[np.arange(len(labels_np)), labels_np].cpu().numpy()

    X_train, X_test, y_train, y_test, covered_train, covered_test = train_test_split(features, labels_np, covered,
                                                                                     test_size=test_fraction,
                                                                                     random_state=random_state)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = calWSC(X_train, covered_train, y_train, delta=delta, M=M,
                                              random_state=random_state, verbose=verbose)
    # Estimate coverage
    coverage = wsc_vab(X_test, y_test, covered_test, v_star, a_star, b_star)
    return coverage.item()


def wsc_vab(featreus, labels, covered, v, a, b):
    z = np.dot(featreus, v)
    idx = (z >= a) & (z <= b)
    return np.mean(covered[idx])


def calWSC(X, y, covered, delta=0.1, M=1000, random_state=2020, verbose=True):
    rng = np.random.default_rng(random_state)
    n = len(y)

    def wsc_v(X, covered, delta, v):
        z = np.dot(X, v)
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = covered[z_order]

        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n - 1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n - 1)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n: int, p: int) -> np.ndarray:
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    results = np.zeros((M, 4))

    iterator = tqdm(range(M)) if verbose else range(M)

    for m in iterator:
        wsc, a, b = wsc_v(X, covered, delta, V[m])
        results[m] = [wsc, a, b, m]

    idx_best = np.argmin(results[:, 0])
    wsc_star = results[idx_best, 0]
    a_star = results[idx_best, 1]
    b_star = results[idx_best, 2]
    v_star = V[int(results[idx_best, 3])]

    return wsc_star, v_star, a_star, b_star


@METRICS_REGISTRY_CLASSIFICATION.register()
def singleton_hit_ratio(prediction_sets, labels):
    if len(prediction_sets) == 0:
        raise AssertionError("The number of prediction set must be greater than 0.")
    n = len(prediction_sets)
    singletons = torch.sum(prediction_sets, dim=1) == 1
    covered = prediction_sets[torch.arange(len(labels)), labels]

    return torch.sum(singletons & covered).item() / n


def compute_p_values(cal_scores, test_scores, smooth=False):
    """
    Compute p-values for conformal prediction.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        Tensor: p-values for each test sample and class, shape (n_test, k)
    """

    n_cal = cal_scores.size(0)
    n_test, k = test_scores.size()

    cal_scores_expanded = cal_scores.view(1, n_cal, 1)
    test_scores_expanded = test_scores.view(n_test, 1, k)

    greater = (cal_scores_expanded > test_scores_expanded).sum(dim=1)
    equal = (cal_scores_expanded == test_scores_expanded).sum(dim=1)

    tau = torch.rand_like(equal, dtype=torch.float)

    if smooth:
        p_values = (greater + tau * (equal + 1)) / (n_cal + 1)
    else:
        p_values = (greater + (equal + 1)) / (n_cal + 1)
    return p_values


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_S(cal_scores, test_scores, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Sum criterion: measures efficiency by the average sum of the p-values.
                Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: The average sum of the p-values across all test samples.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    return p_values.sum(dim=1).mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_N(cal_scores, test_scores, alpha, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Number criterion: uses the average size of the prediction sets.
                    Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        alpha (float): The significance level.
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: The average size of the prediction sets.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    return (p_values > alpha).sum(dim=1).float().mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_U(cal_scores, test_scores, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Unconfidence criterion: uses the average unconfidence over the test sequence, 
                            where the unconfidence for a test object x_i is the second largest p-value.
                            Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Mean of second-largest p-values across test samples.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    n_test = p_values.size(0)

    max_classes = torch.argmax(p_values, dim=1)

    mask = torch.ones_like(p_values, dtype=bool)
    mask[torch.arange(n_test), max_classes] = False

    second_values = torch.where(mask, p_values, -torch.inf).amax(dim=1)
    return second_values.mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_F(cal_scores, test_scores, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Fuzziness criterion: uses the average fuzziness where the fuzziness for a test object x_i is defined 
                        as the sum of all p_values apart from a largest one.
                        Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Mean sum of p-values minus the max p-value per sample.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    return (p_values.sum(dim=1) - p_values.max(dim=1).values).mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_M(cal_scores, test_scores, alpha, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Multiple criterion: uses the percentage of objects x_i in the test sequence 
                        for which the prediction set at significance level is multiple.
                        Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        alpha (float): The significance level.
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Proportion of test samples with prediction set size > 1.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    sizes = (p_values > alpha).sum(dim=1)
    return (sizes > 1).float().mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_E(cal_scores, test_scores, alpha, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Excess criterion: uses the average amount the size of the prediction set exceeds 1.
                    Larger values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        alpha (float): The significance level.
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Mean of (set size - 1), clamped at 0, across test samples.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    sizes = (p_values > alpha).sum(dim=1)
    return torch.clamp(sizes - 1, min=0).float().mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_OU(cal_scores, test_scores, test_labels, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Observed Unconfidence criterion: uses the average observed unconfidence over the test sequence, 
            where the observed unconfidence for a test example (x_i, y_i) is the largest p-value for the false labels.
            Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        test_labels (Tensor): Ground-Truth labels for test samples.
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Mean of the highest p-value among incorrect classes per sample.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    n_test = p_values.size(0)

    mask = torch.ones_like(p_values, dtype=bool)
    mask[torch.arange(n_test), test_labels] = False

    largest_false_values = torch.where(mask, p_values, -torch.inf).amax(dim=1)
    return largest_false_values.mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_OF(cal_scores, test_scores, test_labels, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Observed Fuzziness criterion: uses the average sum of the pvalues for the false labels.
                                Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Mean sum of p-values excluding the true label per test sample.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    n_test = p_values.size(0)

    mask = torch.ones_like(p_values, dtype=bool)
    mask[torch.arange(n_test), test_labels] = False

    sum_wrong_values = (p_values * mask).sum(dim=1)
    return sum_wrong_values.mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_OM(cal_scores, test_scores, test_labels, alpha, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Observed Multiple criterion: uses the percentage of observed multiple predictions in the test sequence, 
            where an observed multiple prediction is defined to be a prediction set including a false label.
            Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        test_labels (Tensor): Ground-Truth labels for test samples.
        alpha (float): The significance level.
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Proportion of test samples where prediction set contains at least one wrong class.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    pred_sets = p_values > alpha

    pred_sets[torch.arange(pred_sets.size(0)), test_labels] = False
    return (pred_sets.sum(dim=1) > 0).float().mean()


@METRICS_REGISTRY_CLASSIFICATION.register()
def pvalue_criterion_OE(cal_scores, test_scores, test_labels, alpha, smooth=False):
    """
    Paper: Criteria of efficiency for conformal prediction (Vovk et al., 2016)

    Observed Excess criterion: uses the average number of false labels included 
                            in the prediction sets at significance level.
                            Smaller values are preferable.

    Args:
        cal_scores (Tensor): Nonconformity scores from the calibration set, shape (n_cal,).
        test_scores (Tensor): Nonconformity scores for test samples across k classes, shape (n_test, k).
        test_labels (Tensor): Ground-Truth labels for test samples.
        alpha (float): The significance level.
        smooth (bool): Whether to apply randomized smoothing when calibration scores equal test scores.

    Returns:
        float: Mean number of wrong classes in prediction sets across test samples.
    """
    p_values = compute_p_values(cal_scores, test_scores, smooth)
    pred_sets = p_values > alpha

    pred_sets[torch.arange(pred_sets.size(0)), test_labels] = False
    return pred_sets.sum(dim=1).float().mean()


class Metrics:

    def __call__(self, metric) -> Any:
        if metric not in METRICS_REGISTRY_CLASSIFICATION.registered_names():
            raise NameError(f"The metric: {metric} is not defined in TorchCP.")
        return METRICS_REGISTRY_CLASSIFICATION.get(metric)
