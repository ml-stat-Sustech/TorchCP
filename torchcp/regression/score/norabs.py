# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim as optim

from torchcp.regression.score.abs import ABS
from torchcp.regression.utils import build_regression_model


import torch

class TorchMinMaxScaler:
    def __init__(self, clip=True):
        self.min_ = None
        self.scale_ = None
        self.clip = clip

    def fit(self, data):
        self.min_ = torch.min(data, dim=0)[0]
        max_ = torch.max(data, dim=0)[0]
        self.scale_ = max_ - self.min_
        # Handle cases where max == min to avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, data):
        if self.min_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        
        min_on_device = self.min_.to(data.device)
        scale_on_device = self.scale_.to(data.device)
        
        normalized_data = (data - min_on_device) / scale_on_device
        if self.clip:
            normalized_data = torch.clamp(normalized_data, 0, 1)
        return normalized_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class DifficultyEstimator:
    """
    Estimates prediction difficulty. It is configured at initialization and
    calibrated with a trained model before application. All internal tensors
    are handled on a specified device.
    """
    def __init__(self, cal_loader, estimator_type='knn_distance',
                 k=10, scalar=True, beta=0.01, custom_function=None, device=None):
        """
        Initializes the configuration for the difficulty estimator.

        Args:
            cal_loader (DataLoader): DataLoader for the calibration dataset.
            estimator_type (str): Method for difficulty estimation.
            k (int): Number of neighbors for k-NN methods.
            scalar (bool): If True, applies min-max scaling.
            beta (float): Small constant for numerical stability.
            custom_function (callable): Custom function for 'function' mode.
            device (torch.device, optional): The device to store tensors on and run
                                             computations. If None, defaults to CPU.
        """
        self.cal_loader = cal_loader
        self.estimator_type = estimator_type
        self.k = k
        self.scalar = scalar
        self.beta = beta
        self.custom_function = custom_function
        self.device = device if device is not None else torch.device("cpu")
        
        self._scaler = None
        self.is_calibrated = False

        supported = ['variance', 'knn_distance', 'knn_label', 'knn_residual', 'function']
        if self.estimator_type not in supported:
            raise ValueError(f"Unsupported estimator_type. Supported: {supported}")
        if self.estimator_type == 'function' and not callable(self.custom_function):
            raise ValueError("A callable `custom_function` must be provided.")

    def calibrate(self, model):
        """
        Calibrates the estimator using a trained model.

        This method extracts calibration data, moves it to the specified device,
        computes necessary statistics, and fits the internal scaler if enabled.

        Args:
            model (torch.nn.Module): The trained regression model.
        """
        if self.is_calibrated:
            return

        model.eval()
        model.to(self.device)

        # 1. Extract all data and get predictions on the correct device
        X_cal_list, y_cal_list, predicts_cal_list = [], [], []
        with torch.no_grad():
            for x, y in self.cal_loader:
                x_dev = x.to(self.device)
                X_cal_list.append(x_dev)
                y_cal_list.append(y.to(self.device))
                predicts_cal_list.append(model(x_dev))
        
        # 2. Store all necessary tensors directly on the target device
        self.X_cal = torch.cat(X_cal_list, dim=0)
        self.y_cal = torch.cat(y_cal_list, dim=0).squeeze()
        predicts_cal = torch.cat(predicts_cal_list, dim=0)

        if 'knn' in self.estimator_type:
            if self.estimator_type == 'knn_residual':
                residuals_cal = self.y_cal - predicts_cal[:, 0].squeeze()
                self.residuals_cal = residuals_cal.to(torch.float32)

        # 3. Fit the scaler if enabled. All computations happen on the target device.
        if self.scalar:
            if self.estimator_type == 'variance':
                raw_cal_scores = predicts_cal[:, 1]
            elif self.estimator_type == 'function':
                raw_cal_scores = self._compute_difficulty(self.X_cal, predicts_cal)
            else:
                raw_cal_scores = self._compute_difficulty(self.X_cal, predicts_batch=None)
            
            self._scaler = TorchMinMaxScaler().fit(raw_cal_scores.unsqueeze(1))
            self._scaler.min_ = self._scaler.min_.to(self.device)
            self._scaler.scale_ = self._scaler.scale_.to(self.device)
        
        self.is_calibrated = True

    def _compute_difficulty(self, x_batch, predicts_batch=None):
        """Internal helper to compute raw difficulty scores on the target device."""
        if self.estimator_type == 'variance':
            if predicts_batch is None or predicts_batch.ndim != 2 or predicts_batch.shape[-1] != 2:
                raise ValueError("For 'variance' mode, `predicts_batch` must have shape (batch_size, 2).")
            return predicts_batch[:, 1]
        elif self.estimator_type == 'function':
            if (x_batch is None) or (predicts_batch is None): raise ValueError("`x_batch` and `predicts_batch` is required for 'function' mode.")
            return self.custom_function(x_batch, predicts_batch)
        else:
            if x_batch is None: raise ValueError(f"`x_batch` is required for '{self.estimator_type}' mode.")
            # All tensors are already on self.device, so this is efficient.
            dists = torch.cdist(x_batch, self.X_cal)
            knn_dists, knn_indices = torch.topk(dists, self.k, dim=1, largest=False)
            if self.estimator_type == 'knn_distance':
                return torch.sum(knn_dists, dim=1)
            elif self.estimator_type == 'knn_label':
                return torch.std(self.y_cal[knn_indices], dim=1)
            elif self.estimator_type == 'knn_residual':
                return torch.mean(torch.abs(self.residuals_cal[knn_indices]), dim=1)

    def apply(self, x_batch, predicts_batch=None):
        """
        Computes difficulty scores for a new batch. Assumes inputs are on the correct device.
        """
        if not self.is_calibrated:
            raise RuntimeError("DifficultyEstimator has not been calibrated. Call `calibrate(model)` first.")

        # The calling function (e.g., SplitPredictor) is responsible for moving
        # x_batch and predicts_batch to the correct device.
        diff_estimate = self._compute_difficulty(x_batch, predicts_batch)

        if self.scalar and self._scaler:
            # `transform` no longer needs to move tensors, as they are already on the same device.
            diff_estimate = self._scaler.transform(diff_estimate.unsqueeze(1)).squeeze(1)

        return diff_estimate + self.beta
    
    
class NorABS(ABS):
    """
    Normalized Absolute Score (NorABS) for conformal regression.

    This score function computes the absolute difference between the prediction
    and the true value, normalized by a difficulty estimate provided by a
    `DifficultyEstimator`.
    """

    def __init__(self, difficulty_estimator):
        """
        Initializes the NorABS score function.

        Args:
            difficulty_estimator (DifficultyEstimator): A pre-configured and fitted
                instance of the DifficultyEstimator class.
        """
        super().__init__()
        if not isinstance(difficulty_estimator, DifficultyEstimator):
            raise TypeError("`difficulty_estimator` must be an instance of DifficultyEstimator.")

        self.difficulty_estimator = difficulty_estimator

    def __call__(self, predicts, y_truth, x_batch=None, model=None, device=None):
        """
        Computes the normalized non-conformity score for each sample.

        This method also triggers the one-time calibration of the internal
        DifficultyEstimator when it is first called within the main
        calibration process.
        
        Args:
            (Same as before, with model and device added for calibration)
        """
        if not self.difficulty_estimator.is_calibrated:
            if model is None:
                raise ValueError("A `model` must be provided to calibrate the DifficultyEstimator.")
            self.difficulty_estimator.calibrate(model, device)

        if y_truth.ndim > 1: y_truth = y_truth.squeeze()
        
        mu = predicts[:, 0] 

        diff_estimate = self.difficulty_estimator.apply(x_batch, predicts)
        
        scores = torch.abs(mu - y_truth) / diff_estimate
        return scores.unsqueeze(1)

    def generate_intervals(self, predicts_batch, q_hat, x_batch):
        """
        Generates prediction intervals using predicted means and standard deviations, 
        scaled by the calibrated threshold :attr:`q_hat`.

        Args:
            predicts_batch (torch.Tensor): Tensor of predicted (mean, std), shape (batch_size, 2).
            q_hat (torch.Tensor): Calibrated threshold values, shape (num_thresholds,).

        Returns:
            torch.Tensor: Prediction intervals, shape (batch_size, num_thresholds, 2),
                        where the last dimension contains lower and upper bounds.
        """
        # breakpoint()
        # if len(predicts_batch.shape) == 2:
        #     predicts_batch = predicts_batch.unsqueeze(1)
        
        # breakpoint()
        diff_estimate = self.difficulty_estimator.apply(x_batch, predicts_batch).unsqueeze(1)
            
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        prediction_intervals[..., 0] = predicts_batch[..., 0] - q_hat.view(1, q_hat.shape[0]) * diff_estimate
        prediction_intervals[..., 1] = predicts_batch[..., 0] + q_hat.view(1, q_hat.shape[0]) * diff_estimate
        return prediction_intervals

    def train(self, train_dataloader, **kwargs):
        """
        Trains the probabilistic regression model to predict both mean and variance.

        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
            **kwargs: Additional keyword arguments for training configuration.
                - model (nn.Module, optional): Custom regression model. If None, defaults to
                                            GaussianRegressionModel.
                - epochs (int, optional): Number of training epochs. Defaults to 100.
                - criterion (nn.Module, optional): Loss function. Defaults to GaussianNLLLoss.
                - lr (float, optional): Learning rate. Defaults to 0.01.
                - optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam.
                - verbose (bool, optional): Whether to print training progress. Defaults to True.

        Returns:
            nn.Module: The trained regression model.
        """
        device = kwargs.get('device', None)
        model = kwargs.get('model',
                           build_regression_model("GaussianRegressionModel")(next(iter(train_dataloader))[0].shape[1], 64,
                                                                              0.5).to(device))
        epochs = kwargs.get('epochs', 100)
        criterion = kwargs.get('criterion', nn.GaussianNLLLoss())
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._basetrain(model, epochs, train_dataloader, criterion, optimizer, verbose)
        return model