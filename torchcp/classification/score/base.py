# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod


class BaseScore(ABC):
    """
    Abstract base class for all score functions.
    """

    # __metaclass__ = ABCMeta

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, logits, labels=None):
        """Virtual method to compute scores for a data pair (x,y).

        Args:
            probs (torch.Tensor): The prediction probabilities.
            label (torch.Tensor): The ground truth label.
        """
        raise NotImplementedError
