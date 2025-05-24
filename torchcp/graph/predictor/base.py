# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torchcp.utils.common import get_device
from torchcp.classification.utils.metrics import Metrics


class BasePredictor(object):
    """
    Abstract base class for all conformal predictors designed for graph data.

    This class serves as a base for implementing conformal prediction algorithms 
    in the context of graph-structured data. It extends the functionality of 
    `BasePredictor` and incorporates additional methods and properties 
    specific to graph prediction tasks.

    Args:
        graph_data (torch_geometric.data.Data): 
            The input graph data in PyG format.
        score_function (callable): 
            A user-defined function that computes the non-conformity score.
        model (torch.nn.Module): 
            A PyTorch model used for predictions on the graph. Defaults to `None`.
    """

    __metaclass__ = ABCMeta

    def __init__(self, graph_data, score_function, model=None, device=None):
        self.score_function = score_function
        self._model = model

        if device is not None:
            self._device = torch.device(device)
        elif model is not None:
            self._device = get_device(model)
        else:
            self.device = graph_data.x.device

        if self._model != None:
            self._model.eval()
            self._model.to(self._device)

        self._graph_data = graph_data.to(self._device)
        self._label_mask = F.one_hot(graph_data.y).bool()
        self._metric = Metrics()

    @abstractmethod
    def calibrate(self, cal_idx, alpha):
        """
        Abstract method to perform calibration on a given calibration set.

        Args:
            cal_idx (torch.Tensor): 
                Indices specifying the samples in the graph data that belong to 
                the calibration set.
            alpha (float): 
                The significance level, a value in the range (0, 1), representing the 
                acceptable error rate for conformal prediction.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, eval_idx):
        """
        Abstract method to make predictions on a given test set.

        This method must be implemented by subclasses to handle the prediction process 
        for conformal prediction. The prediction will be based on the model's 
        outputs and the non-conformity scores, adjusted according to the calibration.

        Args:
            eval_idx (torch.Tensor or list): 
                Indices specifying the samples in the test set on which predictions 
                need to be made. Shape: [num_test_samples].

        Returns:
            list: 
                A list containing prediction sets for the test samples, depending on 
                the specific conformal prediction method implemented.
        """
        raise NotImplementedError

    def _generate_prediction_set(self, scores, q_hat):
        """
        Generate the prediction set with the threshold q_hat.

        Args:
            scores (torch.Tensor): The non-conformity scores of {(x,y_1),..., (x,y_K)}.
            q_hat (torch.Tensor): The calibrated threshold.

        Returns:
            torch.Tensor: A tensor of 0/1 values indicating the prediction set for each example.
        """

        return (scores <= q_hat).int()
