# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from abc import ABCMeta, abstractmethod

from torchcp.classification.utils import ConfCalibrator
from torchcp.classification.utils.metrics import Metrics
from torchcp.utils.common import get_device


class BasePredictor(object):
    """
    Abstract base class for all conformal predictors.
        
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
    
    Attributes:
        score_function (callable): Non-conformity score function.
        _model (torch.nn.Module): The PyTorch model.
        _device (torch.device): The device on which the model is located.
        _metric (Metrics): An instance of the Metrics class.
        _logits_transformation (ConfCalibrator): The logits transformation using Temperature Scaling.
        
    Methods:
        calibrate(cal_dataloader, alpha):
            Virtual method to calibrate the calibration set.
        predict(x_batch):
            Generate prediction sets for the test examples.
        _generate_prediction_set(scores, q_hat):
            Generate the prediction set with the threshold q_hat.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model=None, temperature=1):
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        
        self.score_function = score_function
        self._model = model
        if self._model != None:
            self._model.eval()
        self._device = get_device(model)
        self._metric = Metrics()
        self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(temperature)

    @abstractmethod
    def calibrate(self, cal_dataloader, alpha):
        """
        Virtual method to calibrate the calibration set.

        Args:
            cal_dataloader (torch.utils.data.DataLoader): A dataloader of the calibration set.
            alpha (float): The significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_batch):
        """
        Generate prediction sets for the test examples.
        
        Args:
            x_batch (torch.Tensor): A batch of input.
        """
        raise NotImplementedError

    def _generate_prediction_set(self, scores, q_hat : torch.Tensor):
        """
        Generate the prediction set with the threshold q_hat.

        Args:
            scores (torch.Tensor): The non-conformity scores of {(x,y_1),..., (x,y_K)}.
            q_hat (torch.Tensor): The calibrated threshold.

        Returns:
            list: A list of prediction sets for each example.
        """

        return [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
