# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import math
import warnings
from tqdm import tqdm
import typing

from torchcp.regression.predictor.aci import ACIPredictor

class AgACIPredictor(ACIPredictor):
    def __init__(self, score_function, model, gamma_list, aggregation_function='mean', threshold=[-99999, 99999]):
        super().__init__(score_function, model, None)

        self.gamma_list = gamma_list
        self.alpha_t = None
        self.model_backbone = model
        self.train_indicate = False
        
        if aggregation_function == 'mean':
            self.aggregation_function = torch.mean
        elif aggregation_function == 'median':
            self.aggregation_function = lambda x, dim: torch.median(x, dim=dim)[0]
        else:
            self.aggregation_function = aggregation_function
            
        self.lower_threshold, self.upper_threshold = threshold[0], threshold[1]
        

    def generate_aci_intervals(self, x_batch, x_lookback, y_lookback, pred_interval_lookback, predicts_batch):
        """
        Generate Aggregation Adaptive Conformal Intervals (AgACI) and compute the weighted confidence interval.

        Args:
        - x_batch: Current batch input data
        - x_lookback: Past input data
        - y_lookback: Past ground truth labels (N,)
        - pred_interval_lookback: Past prediction intervals (N, 2) [lower_bound, upper_bound]
        - predicts_batch: Model predictions for the current batch

        Returns:
        - weighted_intervals: The weighted confidence intervals
        """
        err_t = self.calculate_err_rate(x_batch, y_lookback, pred_interval_lookback, weight=False)
        scores = self.calculate_score(self._model(x_lookback).float(), y_lookback)

        intervals_list = []
        weight_list = []

        for gamma in self.gamma_list:
            # Compute the adaptive alpha_t
            alpha_t = max(1 / (scores.shape[0] + 1), min(0.9999, self.alpha + gamma * (self.alpha - err_t)))
            q_hat = self._calculate_conformal_value(scores, alpha_t)
            pred_intervals = self.generate_intervals(predicts_batch, q_hat)
            
            # Ensure pred_intervals are within a valid range
            pred_intervals[:, :, 0] = torch.where(
                torch.isfinite(pred_intervals[:, :, 0]), pred_intervals[:, :, 0], self.lower_threshold
            )
            pred_intervals[:, :, 1] = torch.where(
                torch.isfinite(pred_intervals[:, :, 1]), pred_intervals[:, :, 1], self.upper_threshold
            )

            intervals_list.append(pred_intervals)

            # Compute the Pinball Loss
            quantiles = torch.tensor([self.alpha / 2, 1 - self.alpha / 2], device=y_lookback.device)
            lower_bound, upper_bound = pred_interval_lookback[:, :, 0], pred_interval_lookback[:, :, 1]

            loss_lower = torch.maximum(quantiles[0] * (y_lookback - lower_bound),
                                    (quantiles[0] - 1) * (y_lookback - lower_bound))
            loss_upper = torch.maximum(quantiles[1] * (y_lookback - upper_bound),
                                    (quantiles[1] - 1) * (y_lookback - upper_bound))

            # Compute weighted loss
            weight = torch.tensor([[self.aggregation_function(loss_lower), self.aggregation_function(loss_upper)]],
                                device=y_lookback.device)
            weight_list.append(weight)

        # Compute the weighted confidence interval
        stacked_intervals = torch.stack(intervals_list)  
        stacked_weights = torch.stack(weight_list)

        # Normalize weights and ensure the denominator is nonzero
        weight_sum = torch.sum(stacked_weights, dim=0, keepdim=True)
        weight_sum = torch.where(weight_sum == 0, torch.tensor(1.0, device=weight_sum.device), weight_sum)  # Avoid division by zero

        weighted_intervals = torch.sum(stacked_intervals * stacked_weights[:, None, :], dim=0) / weight_sum
        return weighted_intervals
