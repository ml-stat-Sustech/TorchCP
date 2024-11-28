# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import abstractmethod

import torch.nn.functional as F

from torchcp.graph.utils.metrics import Metrics
from torchcp.classification.predictors.base import BasePredictor

class BaseGraphPredictor(BasePredictor):
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

    def __init__(self, graph_data, score_function, model=None):
        super(BaseGraphPredictor, self).__init__(score_function, model)

        self._graph_data = graph_data
        self._label_mask = F.one_hot(graph_data.y).bool()
        self._device = graph_data.device
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