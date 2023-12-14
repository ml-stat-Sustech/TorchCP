# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta, abstractmethod


class BaseScoreFunction(object):
    """
    Abstract base class for all score functions.

    """
    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, logits, y):
        """Virtual method to compute scores for a data pair (x,y).

        :param logits: the logits for an input.
        :y : the label.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, logits):
        """Virtual method to compute scores of all labels for input x.

        :param logits: the logits for an input.
        """
        raise NotImplementedError
