# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta, abstractmethod

import torch

from torchcp.classification.utils import ConfCalibrator
from torchcp.classification.utils.metrics import Metrics
from torchcp.utils.common import get_device


class BasePredictor(object):
    """
    Abstract base class for all predictors classes.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model=None, temperature=1):
        """
        :param score_function: non-conformity score function.
        :param model: a deep learning model.
        """

        self.score_function = score_function
        self._model = model
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
        Generate prediction sets for  test examples.
        
        :param x_batch: a batch of input.
        """
        raise NotImplementedError

    def _generate_prediction_set(self, scores, q_hat):
        """
        Generate the prediction set with the threshold q_hat.

        :param scores : The non-conformity scores of {(x,y_1),..., (x,y_K)}
        :param q_hat : the calibrated threshold.
        """
        if len(scores.shape) == 1:
            return torch.argwhere(scores < q_hat).reshape(-1).tolist()
        else:
            return [torch.argwhere(scores[i] < q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]

