# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABCMeta, abstractmethod


class BaseScore(object):
    """
    Abstract base class for all graph score functions.

    :param base_score_function: basic non-conformity score funtion.
    :param graph_data: data of graph.
    """
    __metaclass__ = ABCMeta

    def __init__(self, base_score_function, graph_data) -> None:
        self._base_score_function = base_score_function
        self._edge_index = graph_data.edge_index
        self._edge_weight = graph_data.edge_weight
        self._n_vertices = graph_data.num_nodes

    @abstractmethod
    def __call__(self, logits):
        """Virtual method to compute scores for a data pair (x,y).

        :param logits: the logits for inputs.

        """
        raise NotImplementedError
