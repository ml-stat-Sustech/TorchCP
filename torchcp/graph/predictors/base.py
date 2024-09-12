# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from abc import ABCMeta

from torchcp.graph.utils.metrics import Metrics
from torchcp.utils.common import get_device


class BaseGraphPredictor(object):
    """
    Abstract base class for all conformal predictors in graph.

    :param score_function: graph non-conformity score function.
    :param model: a pytorch model.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model=None):
        self.score_function = score_function
        self._model = model
        self._device = get_device(model)
        self._metric = Metrics()

    def _generate_prediction_set(self, scores, q_hat):
        """
        Generate the prediction set with the threshold q_hat.

        :param scores : The non-conformity scores of {(x,y_1),..., (x,y_K)}
        :param q_hat : the calibrated threshold.
        """

        return [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
