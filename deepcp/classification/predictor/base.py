# Copyright (c) 2018-present, ml-stat-Sustech.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta,abstractmethod


class BasePredictor(object):
    """
    Abstract base class for all attack classes.

    :param predict: forward pass function.
    :param loss_fn: loss function that takes .
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    """

    __metaclass__ = ABCMeta

    def __init__(self,score_function):
        """
        :calibration_method: methods used to calibrate 
        :param **kwargs: optional parameters used by child classes.
        """

        self.score_function = score_function

            
    @abstractmethod
    def fit(self, x,y,alpha):
        """Virtual method for generating the adversarial examples.

        :param x: the model's output logits.
        :y : optional parameters used by child classes.
        :alpha: the significance level.
        """
        raise NotImplementedError
        
    @abstractmethod
    def predict(self, x):
        """ prediction sets for test examples.

        :param x: the model's output logits.
        """
        raise NotImplementedError