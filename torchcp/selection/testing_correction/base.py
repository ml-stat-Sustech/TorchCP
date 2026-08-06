# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta, abstractmethod
import torch
from tqdm import tqdm

from torchcp.utils.common import get_device


class Base(object):
    """
    Abstract base class for all multiple testing correction algorithms.
    """
    __metaclass__ = ABCMeta

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, p_values, alpha):
        raise NotImplementedError
