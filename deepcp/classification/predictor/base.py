# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta, abstractmethod


import torch

from deepcp.classification.utils.metrics import Metrics
from deepcp.classification.utils import ConfCalibrator

class BasePredictor(object):
    """
    Abstract base class for all predictor classes.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model= None):
        """
        :score_function: non-conformity score function.
        :param model: a deep learning model.
        """

        self.score_function = score_function
        self._model = model
        if self._model ==  None:
            self._model_device = None
        else:
            self._model_device = next(model.parameters()).device
        self._metric = Metrics()
        self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("Identity")()

    @abstractmethod
    def calibrate(self, cal_dataloader, alpha):
        """Virtual method to calibrate the calibration set.

        :param cal_dataloader : a dataloader of calibration set.
        :param alpha: the significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_batch):
        """generate prediction sets for  test examples.
        :param x_batch: a batch of input.
        """
        raise NotImplementedError
    
        
    
    def _generate_prediction_set(self, scores, q_hat):
        """Generate the prediction set with the threshold q_hat.

        Args:
            scores (_type_): The non-conformity scores of {(x,y_1),..., (x,y_K)}
            q_hat (_type_): the calibrated threshold.

        Returns:
            _type_: _description_
        """
        if len(scores.shape) ==1:
            return torch.argwhere(scores < q_hat).reshape(-1).tolist()
        else:
            return torch.argwhere(scores < q_hat).tolist()
