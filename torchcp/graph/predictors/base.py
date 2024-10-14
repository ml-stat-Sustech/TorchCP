# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F

from torchcp.graph.utils.metrics import Metrics
from torchcp.utils.common import get_device
from torchcp.classification.predictors import BasePredictor


class BaseGraphPredictor(BasePredictor):
    """
    Abstract base class for all conformal predictors in graph.

    :param score_function: graph non-conformity score function.
    :param model: a pytorch model.
    :param graph_data: PyG data of graph.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model=None, graph_data=None):
        super().__init__(score_function, model)

        self._graph_data = graph_data
        if graph_data is not None:
            self._label_mask = F.one_hot(graph_data.y).bool()
        self._device = get_device(model)
        self._metric = Metrics()

    @abstractmethod
    def calibrate(self, cal_idx, alpha):
        """
        Virtual method to calibrate the calibration set.

        :param cal_idx : index of calibration set.
        :param alpha: the significance level.
        """
        raise NotImplementedError
