# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from typing import List

from torchcp.regression.predictor.aci import ACIPredictor

class AgACIPredictor(ACIPredictor):
    """
    Online Expert Aggregation Adaptive Conformal Inference.
    
    A parameter-free method that adaptively builds upon ACI based on online expert aggregation.
    
    Args:
        score_function (torchcp.regression.scores): A class that implements the score function.
        model (torch.nn.Module): A PyTorch model capable of outputting quantile values.
            The model should be an initialization model that has not been trained.
        gamma_list (List[float]): A list of step size parameters for adaptive adjustment of alpha at each step.
            Each element in the list corresponds to a different expert. Must contain values greater than 0.
        aggregation_function (str or callable, optional): The function used to aggregate predictions from experts.
            Can be either 'mean' (average), 'median', or a custom callable function. Defaults to 'mean'.
        threshold (List[float], optional): A list containing the lower and upper thresholds for clipping expert predictions.
            Defaults to [-99999, 99999] (effectively no clipping).
        
    Reference:  
        Paper: Adaptive Conformal Predictions for Time Series (Zaffran et al., 2022)
        Link: https://proceedings.mlr.press/v162/zaffran22a.html
        Github: https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries
        
    """
    
    def __init__(self, score_function, model, gamma_list: List, aggregation_function='mean', threshold: List=[-99999, 99999]):
        super().__init__(score_function, model, None)
        if aggregation_function not in ['mean', 'median'] and not callable(aggregation_function):
            raise ValueError(
                "aggregation_function must be either 'mean', 'median', or a callable function."
            )
        if not isinstance(gamma_list, list):
            raise ValueError(f"gamma_list must be a list, but got {type(gamma_list).__name__}.")
        if not isinstance(threshold, list) or len(threshold) != 2:
            raise ValueError(f"threshold must be a list with exactly 2 elements, but got {threshold}.")
        

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
        y_lookback = y_lookback.unsqueeze(1)
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
                                    (quantiles[0] - 1) * (y_lookback - lower_bound)).detach()
            loss_upper = torch.maximum(quantiles[1] * (y_lookback - upper_bound),
                                    (quantiles[1] - 1) * (y_lookback - upper_bound)).detach()

            # Compute weighted loss
            # breakpoint()
            weight = torch.tensor([[self.aggregation_function(loss_lower, dim=0).tolist(), self.aggregation_function(loss_upper, dim=0).tolist()]],
                                device=y_lookback.device) 
            weight_list.append(weight)
        # breakpoint()
        # Compute the weighted confidence interval
        stacked_intervals = torch.stack(intervals_list)   # (num_experts, x_batch_size, 1, 2)
        stacked_weights = torch.stack(weight_list).permute(0, 3, 1, 2)  
        
        # Normalize weights and ensure the denominator is nonzero
        weight_sum = torch.sum(stacked_weights, dim=0, keepdim=True)
        weight_sum = torch.where(weight_sum == 0, torch.tensor(1.0, device=weight_sum.device), weight_sum)  # Avoid division by zero
        weighted_intervals = torch.sum(stacked_intervals * stacked_weights, dim=0) / weight_sum
        # breakpoint()
        return weighted_intervals[0]