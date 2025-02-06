# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta, abstractmethod

import torch


class BaseScore(object):
    """
    Abstract base class for all graph score functions.

    The class takes in a basic non-conformity score function (to be defined in subclasses) 
    and graph data (e.g., node features, edge information) for computing scores on graph data.

    Args:
        graph_data (torch_geometric.data.Data): 
            The graph data used for computing the scores. It includes information like the number 
            of nodes (`num_nodes`), edge connections (`edge_index`), and edge weights (`edge_weight`).
            This data is typically represented as a `Data` object from PyTorch Geometric.
            
        base_score_function (function): 
            A function that calculates the base non-conformity score. This function will be 
            used in subclasses to define the behavior of the score function.
    """
    __metaclass__ = ABCMeta

    def __init__(self, graph_data, base_score_function) -> None:
        self._base_score_function = base_score_function

        self._n_vertices = graph_data.num_nodes
        edge_index = graph_data.edge_index
        edge_weight = graph_data.edge_weight
        self._device = edge_index.device

        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.shape[1]).to(self._device)
        self._adj = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            (self._n_vertices, self._n_vertices))
        self._degs = torch.matmul(self._adj, torch.ones(
            (self._adj.shape[0])).to(self._device))

    @abstractmethod
    def __call__(self, logits, labels=None):
        """
        Virtual method to compute non-conformity scores for a data pair (x, y).

        This method must be implemented by subclasses to compute the non-conformity score 
        based on the logits (predicted values) for the input data.

        Args:
            logits (torch.Tensor): 
                The logits (model output before softmax) for each node. 
                Shape: [num_nodes, num_classes].

            labels (torch.Tensor):
                The labels for each node.
                Shape: [num_nodes]

        Returns:
            torch.Tensor: A tensor containing the non-conformity scores for each data sample. 
                          The shape will depend on the specific implementation.
        """
        raise NotImplementedError
