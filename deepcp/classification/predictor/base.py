# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta, abstractmethod


class BasePredictor(object):
    """
    Abstract base class for all predictor classes.

    :param score_function: non-conformity score function.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function):
        """
        :calibration_method: methods used to calibrate 
        :param **kwargs: optional parameters used by child classes.
        """

        self.score_function = score_function

    @abstractmethod
    def calibrate(self, x, y, alpha):
        """Virtual method to calibrate the calibration set.

        :param x: the model's output logits.
        :y : labels of calibration set.
        :alpha: the significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """generate a prediction set for a test example.

        :param x: the model's output logits.
        """
        raise NotImplementedError
