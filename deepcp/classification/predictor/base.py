# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta, abstractmethod


import numpy as np
import torch
from tqdm import tqdm

from deepcp.classification.utils.metircs import Metrics
from deepcp.classification.utils import ConfCalibrator

class BasePredictor(object):
    """
    Abstract base class for all predictor classes.

    :param score_function: non-conformity score function.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model):
        """
        :calibration_method: methods used to calibrate 
        :param **kwargs: optional parameters used by child classes.
        """

        self.score_function = score_function
        self._model = model
        if self._model ==  None:
            self._model_device = None
        else:
            self._model_device = next(model.parameters()).device
        self._metric = Metrics()
        self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("Identity")

    @abstractmethod
    def calibrate(self,model, cal_dataloader, alpha):
        """Virtual method to calibrate the calibration set.

        :param model: the deep learning model.
        :param cal_dataloader : dataloader of calibration set.
        :param alpha: the significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """generate a prediction set for a test example.

        :param x: the model's output logits.
        """
        raise NotImplementedError
    
        
    
    def _generate_prediction_set(self,scores, q_hat):
        """Generate the prediction set with the threshold q_hat.

        Args:
            scores (_type_): The non-conformity scores of {(x,y_1),..., (x,y_K)}
            q_hat (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.argwhere(scores < q_hat).reshape(-1).tolist()
