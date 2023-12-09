# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta, abstractmethod


class DaseScoreFunction(object):
    """
    Abstract base class for all score functions.

    """
    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, probs, y):
        """Virtual method to compute scores for a data pair (x,y).

        :param probs: the model's output probs for an input.
        :y : the label.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, probs):
        """Virtual method to compute scores of all labels for input x.

        :param probs: the model's output probabilities for an input.
        """
        raise NotImplementedError
