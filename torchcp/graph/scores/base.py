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
    """
    __metaclass__ = ABCMeta

    def __init__(self, base_score_function) -> None:
        self._base_score_function = base_score_function

    @abstractmethod
    def __call__(self, logits, n_vertices, edge_index, edge_weights=None, knn_edge=None, knn_weights=None):
        """Virtual method to compute scores for a data pair (x,y).

        :param base_scores: the basic scores for inputs.
        :param edge_index: the edge indices or the adjacency matrix.
        :param edge_weights: the weights corresponding to the edges.
        :param knn_edge: the edge indices of the similarity-based graph, such as knn.
        :param knn_weights: the edge weights of the similarity-based graph, such as knn.
        """
        raise NotImplementedError
