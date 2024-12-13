# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim as optim

from .base import BasePredictor


class SplitPredictor(BasePredictor):
    """
    Split Conformal Prediction for Regression.
    
    This predictor allows for the construction of a prediction band for the response 
    variable using any estimator of the regression function.
        
    Args:
        score_function (torchcp.regression.scores): A class that implements the score function.
        model (torch.nn.Module): A pytorch regression model that can output predicted point.
        
    Reference:
        Paper: Distribution-Free Predictive Inference For Regression (Lei et al., 2017)
        Link: https://arxiv.org/abs/1604.04173
        Github: https://github.com/ryantibs/conformal
    """

    def __init__(self, score_function, model=None):
        super().__init__(score_function, model)

    def train(self, train_dataloader, **kwargs):
        """
        Trains the model using the provided train_dataloader and score_function.

        Args:
            train_dataloader (DataLoader): DataLoader for the training data.
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
        model = kwargs.pop('model', None)

        if model is not None:
            self._model = self.score_function.train(
                train_dataloader, model=model, device=self._device, **kwargs
            )
        elif self._model is not None:
            self._model = self.score_function.train(
                train_dataloader, model=self._model, device=self._device, **kwargs
            )
        else:
            raise ValueError("No model provided and self._model is not set. Please provide a model or set self._model before training.")


    def calculate_score(self, predicts, y_truth):
        """
        Calculate the nonconformity scores based on the model's predictions and true values.

        Args:
            predicts (torch.Tensor): Model predictions.
            y_truth (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed scores for each prediction.
        """
        return self.score_function(predicts, y_truth)

    def generate_intervals(self, predicts_batch, q_hat):
        """
        Generate prediction intervals based on the model's predictions and the conformal value.

        Args:
            predicts_batch (torch.Tensor): Batch of predictions from the model.
            q_hat (float): Conformal value computed during calibration.

        Returns:
            torch.Tensor: Prediction intervals.
        """
        return self.score_function.generate_intervals(predicts_batch, q_hat)

    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        predicts_list, y_truth_list = [], []
        with torch.no_grad():
            for tmp_x, tmp_labels in cal_dataloader:
                tmp_x, tmp_labels = tmp_x.to(self._device), tmp_labels.to(self._device)
                tmp_predicts = self._model(tmp_x).detach()
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)

        predicts = torch.cat(predicts_list).float().to(self._device)
        y_truth = torch.cat(y_truth_list).to(self._device)
        self.scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(self.scores, alpha)

    def predict(self, x_batch):
        self._model.eval()
        x_batch.to(self._device)
        with torch.no_grad():
            predicts_batch = self._model(x_batch)
            return self.generate_intervals(predicts_batch, self.q_hat)

    def evaluate(self, data_loader):
        y_list, predict_list = [], []
        with torch.no_grad():
            for tmp_x, tmp_y in data_loader:
                tmp_x, tmp_y = tmp_x.to(self._device), tmp_y.to(self._device)
                tmp_prediction_intervals = self.predict(tmp_x)
                y_list.append(tmp_y)
                predict_list.append(tmp_prediction_intervals)

        predicts = torch.cat(predict_list, dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)

        res_dict = {
            "Coverage_rate": self._metric('coverage_rate')(predicts, test_y),
            "Average_size": self._metric('average_size')(predicts)
        }
        return res_dict
