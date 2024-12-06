# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from .split import SplitPredictor


class ACIPredictor(SplitPredictor):
    """
    Adaptive Conformal Inference.
    
    methods for forming prediction sets in an online setting where the data generating 
    distribution is allowed to vary over time in an unknown fashion.
    
    Args:
        model (torch.nn.Module): A PyTorch model capable of outputting quantile values.
        score_function (torchcp.regression.scores): A class that implements the score function.
        gamma (float): Step size parameter for adaptive adjustment of alpha. Must be greater than 0.
        
    Reference:  
        Paper: Adaptive conformal inference Under Distribution Shift (Gibbs et al., 2021)
        Link: https://arxiv.org/abs/2106.00170
        Github: https://github.com/isgibbs/AdaptiveConformal
        
    """

    def __init__(self, model, score_function, gamma):
        super().__init__(score_function, model)
        if gamma <= 0:
            raise ValueError("gamma must be greater than 0.")

        self.gamma = gamma
        self.alpha_t = None

    def train(self, train_dataloader, alpha, **kwargs):
        """
        Train and calibrate the predictor using the training data.

        Args:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            alpha (float): Desired initial coverage rate.
            **kwargs: Additional keyword arguments for training configuration.
                - model (nn.Module, optional): The model to be trained.
                - epochs (int, optional): Number of training epochs.
                - criterion (nn.Module, optional): Loss function.
                - lr (float, optional): Learning rate for the optimizer.
                - optimizer (torch.optim.Optimizer, optional): Optimizer for training.
                - verbose (bool, optional): If True, prints training progress.
            
        .. note::
            This function is optional but recommended, because the training process for each score_function is different. 
            We provide a default training method, and users can change the hyperparameters :attr:`kwargs` to modify the training process.
            If the train function is not used, users should pass the trained model to the predictor at the beginning.
        """
        super().train(train_dataloader, alpha=alpha, **kwargs)
        super().calibrate(train_dataloader, alpha)
        self.alpha = alpha
        self.alpha_t = alpha

    def calculate_err_rate(self, x_batch, y_batch_last, pred_interval_last):
        """
        Calculate the error rate for the previous prediction intervals.

        Args:
            x_batch (torch.Tensor): Input features for the current batch.
            y_batch_last (torch.Tensor): True labels from the previous step.
            pred_interval_last (torch.Tensor): Prediction intervals from the previous step.

        Returns:
            float: Weighted error rate based on historical predictions.
        """
        steps_t = len(y_batch_last)
        w_s = (steps_t - torch.arange(steps_t)).to(self._device)
        w_s = torch.pow(0.95, w_s)
        w_s = w_s / torch.sum(w_s)
        err = x_batch.new_zeros(steps_t, self.q_hat.shape[0])
        err = ((y_batch_last >= pred_interval_last[..., 0, 1]) | (y_batch_last <= pred_interval_last[..., 0, 0])).int()
        err_t = torch.sum(w_s * err)
        return err_t

    def predict(self, x_batch, x_batch_last=None, y_batch_last=None, pred_interval_last=None):
        """
        Generate prediction intervals for the input batch.

        Args:
            x_batch (torch.Tensor): Input features for the current batch.
            x_batch_last (torch.Tensor): Input features from the last step.
            y_batch_last (torch.Tensor, optional): Labels from the last step. Defaults to None.
            pred_interval_last (torch.Tensor, optional): Previous prediction intervals. Defaults to None.

        Returns:
            torch.Tensor: Prediction intervals for the input batch.
        """
        if not ((x_batch_last is None) == (y_batch_last is None) == (pred_interval_last is None)):
            raise ValueError("x_batch_last, y_batch_last and pred_interval_last must either be provided or be None.")
        self._model.eval()
        x_batch = x_batch.to(self._device)

        if y_batch_last is None:
            err_t = self.alpha
        else:
            err_t = self.calculate_err_rate(x_batch, y_batch_last, pred_interval_last)
            self.scores = self.calculate_score(self._model(x_batch).float(), y_batch_last)

        self.alpha_t = max(0.0001, min(0.9999, self.alpha_t + self.gamma * (self.alpha - err_t)))

        self.q_hat = self._calculate_conformal_value(self.scores, self.alpha_t)
        predicts_batch = self._model(x_batch.to(self._device)).float()
        return self.generate_intervals(predicts_batch, self.q_hat)

    def evaluate(self, data_loader, verbose=True):
        """
        Evaluate the predictor on a dataset.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for evaluation data.
            verbose (bool, optional): Whether to print evaluation metrics. Defaults to True.

        Returns:
            dict: Dictionary containing evaluation metrics:
                  - Total batches
                  - Average coverage rate
                  - Average prediction interval size
        """
        coverage_rates = []
        average_sizes = []

        with torch.no_grad():
            x_batch_last = None
            y_batch_last = None
            pred_interval_last = None
            for index, batch in enumerate(data_loader):
                x_batch, y_batch = batch[0].to(self._device), batch[1].to(self._device)
                prediction_intervals = self.predict(x_batch, x_batch_last, y_batch_last, pred_interval_last)
                x_batch_last = x_batch
                y_batch_last = y_batch
                pred_interval_last = prediction_intervals

                batch_coverage_rate = self._metric('coverage_rate')(prediction_intervals, y_batch)
                batch_average_size = self._metric('average_size')(prediction_intervals)

                if verbose:
                    print(
                        f"Batch: {index + 1}, Coverage rate: {batch_coverage_rate:.4f}, Average size: {batch_average_size:.4f}, Alpha: {self.alpha_t:.2f}")

                coverage_rates.append(batch_coverage_rate)
                average_sizes.append(batch_average_size)

        avg_coverage_rate = sum(coverage_rates) / len(coverage_rates)
        avg_average_size = sum(average_sizes) / len(average_sizes)

        res_dict = {"Total batches": index + 1,
                    "Coverage_rate": avg_coverage_rate,
                    "Average_size": avg_average_size}

        return res_dict
