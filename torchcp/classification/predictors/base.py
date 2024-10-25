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
        
    :param score_function: non-conformity score function.
    :param model: a pytorch model.
    :param temperature: the temperature of Temperature Scaling.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model=None, temperature=1):
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

        :param cal_dataloader : a dataloader of calibration set.
        :param alpha: the significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_batch):
        """
        Generate prediction sets for the test examples.
        
        :param x_batch: a batch of input.
        """
        raise NotImplementedError

    def _generate_prediction_set(self, scores, q_hat : torch.Tensor):
        """
        Generate the prediction set with the threshold q_hat.

        :param scores : The non-conformity scores of {(x,y_1),..., (x,y_K)}
        :param q_hat : the calibrated threshold.
        """

        return [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
