# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch.nn as nn
from abc import ABCMeta, abstractmethod

from torchcp.utils.common import get_device
from torchcp.utils.common import calculate_conformal_value
from ..utils.metrics import Metrics

class BasePredictor(object):
    """
    Abstract base class for all conformal predictors.
        
    Args:
        score_function: non-conformity score function.
        model: a pytorch model.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model=None):
        self._model = model
        if self._model is not None:
            if not isinstance(model, nn.Module):
                raise TypeError("The model must be an instance of torch.nn.Module")
            self._device = get_device(model)
        else:
            self._device = None
        self.score_function = score_function
        self._metric = Metrics()
        
    @abstractmethod
    def train(self, train_dataloader, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def calculate_score(self, predicts, y_truth):
        raise NotImplementedError
    
    @abstractmethod
    def generate_intervals(self, predicts_batch, q_hat):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_batch):
        """
        Generates prediction intervals for a batch of input data.

        Args:
            x_batch (torch.Tensor): Input batch of data points, shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Prediction intervals, shape (batch_size, 2).
        """
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, data_loader):
        """
        Evaluate the model on a test dataloader, returning coverage rate and interval size.
        
        Args:
            data_loader (torch.utils.data.DataLoader): The dataloader containing the test dataset.
            
        Returns:
            dict: A dictionary containing the coverage rate and average interval size with keys:
            - Coverage_rate (float): The coverage rate of the prediction intervals.
            - Average_size (float): The average size of the prediction intervals.
        """
        raise NotImplementedError

    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    def calibrate(self, cal_dataloader, alpha):
        """
        Calibrate the predictor using a calibration dataset and a given significance level :attr:`alpha`.
        
        Args:
            cal_dataloader (torch.utils.data.DataLoader): The dataloader containing the calibration dataset.
            alpha (float): The significance level for calibration. Should be in the range (0, 1).
        """
        raise NotImplementedError