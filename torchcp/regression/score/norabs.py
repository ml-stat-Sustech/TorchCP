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


class TorchMinMaxScaler:
    """
    A PyTorch implementation of Min-Max normalization with optional clipping.
    Used to normalize difficulty estimates or residuals into [0, 1].
    """
    def __init__(self, clip=True):
        self.min_ = None
        self.scale_ = None
        self.clip = clip

    def fit(self, data):
        """
        Fits the scaler by computing per-feature minimum and maximum.

        Args:
            data (Tensor): Input tensor to normalize.

        Returns:
            TorchMinMaxScaler: The fitted scaler instance.
        """
        self.min_ = torch.min(data, dim=0)[0]
        max_ = torch.max(data, dim=0)[0]
        self.scale_ = max_ - self.min_
        # Avoid division by zero in case of constant features
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, data):
        """
        Applies the fitted transformation to new data.

        Args:
            data (Tensor): Input tensor to normalize.

        Returns:
            Tensor: Normalized tensor in [0, 1] (if clipping enabled).
        """
        if self.min_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        
        min_on_device = self.min_.to(data.device)
        scale_on_device = self.scale_.to(data.device)
        
        normalized_data = (data - min_on_device) / scale_on_device
        if self.clip:
            normalized_data = torch.clamp(normalized_data, 0, 1)
        return normalized_data

    def fit_transform(self, data):
        """Fits the scaler on data and applies the normalization in one step."""
        self.fit(data)
        return self.transform(data)


class DifficultyEstimator:
    """
    Module for estimating sample-specific difficulty scores in conformal regression.

    The estimator can use several strategies (variance-based, k-NN based, 
    residual-based, or a custom function). It requires calibration with a 
    trained regression model before being applied to unseen samples.
    """
    def __init__(self, cal_loader, estimator_type='knn_distance',
                 k=10, scalar=True, beta=0.01, custom_function=None, device=None):
        """
        Initializes the difficulty estimator.

        Args:
            cal_loader (DataLoader): Calibration dataset.
            estimator_type (str): Strategy for difficulty estimation.
                                  Options: 'variance', 'knn_distance', 
                                           'knn_label', 'knn_residual', 'function'.
            k (int): Number of neighbors for k-NN based methods.
            scalar (bool): If True, normalizes raw difficulty scores with min-max scaling.
            beta (float): Additive constant for numerical stability.
            custom_function (callable): User-defined function for 'function' mode.
            device (torch.device, optional): Target device. Defaults to CPU.
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
        Calibrates the estimator using predictions from a trained regression model.

        Extracts calibration data, computes required statistics, 
        and initializes the internal scaling mechanism if enabled.

        Args:
            model (nn.Module): Trained regression model.
        """

        model.eval()
        model.to(self.device)

        # 1. Gather calibration data and predictions
        X_cal_list, y_cal_list, predicts_cal_list = [], [], []
        with torch.no_grad():
            for x, y in self.cal_loader:
                x_dev = x.to(self.device)
                X_cal_list.append(x_dev)
                y_cal_list.append(y.to(self.device))
                predicts_cal_list.append(model(x_dev))
        
        # 2. Store tensors on the target device
        self.X_cal = torch.cat(X_cal_list, dim=0)
        self.y_cal = torch.cat(y_cal_list, dim=0).squeeze()
        predicts_cal = torch.cat(predicts_cal_list, dim=0)

        if 'knn' in self.estimator_type:
            if self.estimator_type == 'knn_residual':
                residuals_cal = self.y_cal - predicts_cal[:, 0].squeeze()
                self.residuals_cal = residuals_cal.to(torch.float32)

        # 3. Fit normalization scaler if required
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
        """
        Internal helper to compute raw difficulty scores on the device.

        Depending on the estimator type, different metrics are computed.
        """
        if self.estimator_type == 'variance':
            if predicts_batch is None or predicts_batch.ndim != 2 or predicts_batch.shape[-1] != 2:
                raise ValueError("For 'variance' mode, `predicts_batch` must have shape (batch_size, 2).")
            return predicts_batch[:, 1]
        elif self.estimator_type == 'function':
            if (x_batch is None) or (predicts_batch is None): 
                raise ValueError("Both `x_batch` and `predicts_batch` are required for 'function' mode.")
            return self.custom_function(x_batch, predicts_batch)
        else:
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
        Applies the calibrated estimator to compute difficulty scores for new data.

        Args:
            x_batch (Tensor): Input features.
            predicts_batch (Tensor, optional): Model predictions.

        Returns:
            Tensor: Difficulty estimates for the batch.
        """
        if not self.is_calibrated:
            raise RuntimeError("DifficultyEstimator must be calibrated via `calibrate(model)` before use.")

        diff_estimate = self._compute_difficulty(x_batch, predicts_batch)

        if self.scalar and self._scaler:
            diff_estimate = self._scaler.transform(diff_estimate.unsqueeze(1)).squeeze(1)

        return diff_estimate + self.beta
    
    
class NorABS(ABS):
    """
    NorABS: Normalized Absolute Score for conformal regression.

    This score computes absolute prediction errors normalized by 
    a calibrated difficulty estimate. It provides adaptive conformity 
    scores that account for local data complexity.
    
    Reference:
    - Inductive Confidence Machines for Regression
    - Reliable prediction intervals with regression neural networks
    - Guaranteed coverage prediction intervals with Gaussian process regression
    """

    def __init__(self, data_loader, estimate_type='variance', k=20, scalar=True, beta=0.01, device=None, custom_function=None):
        """
        Initializes the NorABS score function with its associated DifficultyEstimator.

        Args:
            data_loader (DataLoader): Calibration dataset.
            estimate_type (str): Difficulty estimation strategy.
            k (int): Number of neighbors for k-NN methods.
            scalar (bool): Whether to normalize difficulty estimates.
            beta (float): Stability constant.
            device (torch.device, optional): Target device.
            custom_function (callable, optional): Custom difficulty function.
        """
        super().__init__()
        self.difficulty_estimator = DifficultyEstimator(
            cal_loader=data_loader,
            estimator_type=estimate_type,
            k=k,
            scalar=scalar,
            beta=beta,
            device=device,
            custom_function=custom_function
        )
        
    def calibrate(self, model):
        """Calibrates the internal DifficultyEstimator with a trained model."""
        self.difficulty_estimator.calibrate(model=model)
        

    def __call__(self, predicts, y_truth, x_batch=None, model=None, device=None):
        """
        Computes normalized nonconformity scores for conformal prediction.

        Args:
            predicts (Tensor): Model predictions (mean, std).
            y_truth (Tensor): Ground truth labels.
            x_batch (Tensor, optional): Input features for difficulty estimation.
            model (nn.Module, optional): Regression model (needed for first-time calibration).
            device (torch.device, optional): Target device for calibration.

        Returns:
            Tensor: Normalized nonconformity scores, shape (batch_size, 1).
        """
        if not self.difficulty_estimator.is_calibrated:
            if model is None:
                raise ValueError("A `model` must be provided to calibrate the DifficultyEstimator.")
            self.difficulty_estimator.calibrate(model)
        
        mu = predicts[:, 0] 
        diff_estimate = self.difficulty_estimator.apply(x_batch, predicts)
        
        scores = torch.abs(mu - y_truth) / diff_estimate
        return scores.unsqueeze(1)

    def generate_intervals(self, predicts_batch, q_hat, x_batch):
        """
        Generates prediction intervals based on calibrated conformity scores.

        Args:
            predicts_batch (Tensor): Predicted (mean, std), shape (batch_size, 2).
            q_hat (Tensor): Quantile thresholds, shape (num_thresholds,).
            x_batch (Tensor): Input features for difficulty estimation.

        Returns:
            Tensor: Prediction intervals of shape (batch_size, num_thresholds, 2).
        """
        diff_estimate = self.difficulty_estimator.apply(x_batch, predicts_batch).unsqueeze(1)
            
        prediction_intervals = predicts_batch.new_zeros((predicts_batch.shape[0], q_hat.shape[0], 2))
        prediction_intervals[..., 0] = predicts_batch[..., 0] - q_hat.view(1, q_hat.shape[0]) * diff_estimate
        prediction_intervals[..., 1] = predicts_batch[..., 0] + q_hat.view(1, q_hat.shape[0]) * diff_estimate
        return prediction_intervals

    def train(self, train_dataloader, **kwargs):
        """
        Trains a probabilistic regression model to predict both mean and variance.

        Args:
            train_dataloader (DataLoader): Training dataset.
            **kwargs: Training configuration, including:
                - model (nn.Module, optional): Custom regression model. Defaults to GaussianRegressionModel.
                - epochs (int): Training epochs. Default = 100.
                - criterion (nn.Module): Loss function. Default = GaussianNLLLoss.
                - lr (float): Learning rate. Default = 0.01.
                - optimizer (torch.optim.Optimizer): Optimizer. Default = Adam.
                - verbose (bool): Whether to log progress. Default = True.

        Returns:
            nn.Module: The trained regression model.
        """
        device = kwargs.get('device', None)
        model = kwargs.get(
            'model',
            build_regression_model("GaussianRegressionModel")(
                next(iter(train_dataloader))[0].shape[1], 64, 0.5
            ).to(device)
        )
        epochs = kwargs.get('epochs', 100)
        criterion = kwargs.get('criterion', nn.GaussianNLLLoss())
        lr = kwargs.get('lr', 0.01)
        optimizer = kwargs.get('optimizer', optim.Adam(model.parameters(), lr=lr))
        verbose = kwargs.get('verbose', True)

        self._basetrain(model, epochs, train_dataloader, criterion, optimizer, verbose)
        return model
