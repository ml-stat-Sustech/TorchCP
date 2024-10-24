# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import pandas as pd
import networkx as nx
from scipy.optimize import brentq

from torchcp.classification.scores import APS
from .split import GraphSplitPredictor

DEFAULT_SCHEMES = ["unif", "linear", "geom"]

class NAPSSplitPredictor(GraphSplitPredictor):
    """
    Method: Neighbourhood Adaptive Prediction Sets
    Paper: Distribution Free Prediction Sets for Node Classification (Clarkson et al., 2023)
    Link: https://proceedings.mlr.press/v202/clarkson23a/clarkson23a.pdf
    Github: https://github.com/jase-clarkson/graph_cp/tree/master

    :param G: network of test graph data.
    :param cutoff: nodes with at least 'cutoff' k-hop neighbors for test.
    :param k: Add nodes up to the 'k-hop' neighbors of ego node to its calibration set.
    :param scheme: name of weight decay rate for k-hop neighbors. Options are 'unif' (weights = 1), 'linear' (weights = 1/k), or 'geom' (weights = 2^{-(k - 1)}).
    """

    def __init__(self, G, cutoff=50, k=2, scheme="unif"):
        super().__init__(score_function=APS(score_type="identity"), model =None )

        if scheme not in ["unif", "linear", "geom"]:
            raise ValueError(f"Invalid scheme: {scheme}. Choose from {DEFAULT_SCHEMES}")

        self._cutoff = cutoff
        self._G = G
        self._k = k
        self._scheme = scheme

    def precompute_naps_sets(self, probs, labels, alpha):
        """
        :param probs: predicted probability.
        :param labels: label of node.
        :param alpha: pre-defined empirical marginal coverage 1 - alpha.

        :return lcc_nodes: nodes with at least 'cutoff' k-hop neighbors for test
        :return prediction_sets: lcc_nodes' prediction sets.
        """

        self._device = probs.device
        quantiles_nb = {}
        for node in list(self._G.nodes):
            p = self.calibrate_nbhd(node, probs, labels, alpha)
            if p is not None:
                quantiles_nb.update(p)

        lcc_nodes = torch.tensor(list(quantiles_nb.keys()), device=self._device)
        quantiles = torch.tensor(list(quantiles_nb.values()), device=self._device)
        prediction_sets = self.predict(probs[lcc_nodes], quantiles[:, None])
        return lcc_nodes, prediction_sets

    def calibrate_nbhd(self, node, probs, labels, alpha):
        node_ids, weights = self._get_nbhd_weights(node)

        if self._cutoff <= node_ids.shape[0]:
            quantile = self._calibrate_weighted(probs[node_ids], labels[node_ids],
                                               weights, alpha)
            return {node: quantile}
        return None

    def _get_nbhd_weights(self, node):
        # Get dict containing nodes -> shortest path to node (i.e. depth).
        neigh_depth = nx.single_source_shortest_path_length(self._G, node, cutoff=self._k)
        neigh_depth.pop(node, None) # Remove the node itself from list.
        neigh_count = len(neigh_depth)

        node_ids = torch.tensor(list(neigh_depth.keys()), device=self._device)

        if self._scheme == 'unif':
            weights = torch.ones((neigh_count, ), device=self._device)
        elif self._scheme == 'linear':
            weights = 1. / torch.tensor(list(neigh_depth.values()), device=self._device)
        elif self._scheme == 'geom':
            weights = (0.5)**(torch.tensor(list(neigh_depth.values()), device=self._device) - 1)

        return node_ids, weights
        
    def _calibrate_weighted(self, probs, labels, weights, alpha):
        if probs.shape[0] == 0:
            return alpha
        # Calibrate
        
        alpha_max = 1 - self.score_function(probs, labels)
        scores = alpha - alpha_max
        alpha_correction = self._get_weighted_quantile(scores, weights, alpha)
        return alpha - alpha_correction
    
    def _get_weighted_quantile(self, scores, weights, alpha):
        wtildes = weights / (weights.sum() + 1)
        def critical_point_quantile(q): return (wtildes * (scores <= q)).sum().item() - (1 - alpha)
        try:
            q = brentq(critical_point_quantile, -1000, 1000)
        except ValueError:
            q = 0
            raise ValueError("Did not find a suitable alpha value, keeping alpha unchanged.")
        return q
    
    def predict(self, probs, alphas, allow_empty = True):
        # scores = self.score_function(probs)
        # S = []
        # for index in range(scores.shape[0]):
        #     S.extend(self._generate_prediction_set(scores[index,:].reshape(1,-1), 1-quantiles[index]))
        
        n = probs.shape[0]
        eps = torch.rand(n, device=self._device)

        order = torch.argsort(-probs, dim=1).to(self._device)
        prob_sort = -torch.sort(-probs, dim=1).values
        Z = torch.cumsum(prob_sort, dim=1)

        L = torch.argmax((Z >= 1.0 - alphas).float(), dim=1).flatten()
        Z_excess = Z[torch.arange(n), L] - (1.0 - alphas).flatten()
        p_remove = Z_excess / prob_sort[torch.arange(n), L]
        remove = eps <= p_remove
        for i in torch.where(remove)[0]:
            if not allow_empty:
                L[i] = torch.maximum(torch.tensor(0), L[i] - 1)
            else:
                L[i] = L[i] - 1
        
        S = [order[i, torch.arange(0, L[i] + 1)] for i in range(n)]
            
        return S