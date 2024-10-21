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

from .utils import ProbabilityAccumulator

DEFAULT_SCHEMES = ["unif", "linear", "geom"]

class NAPSSplitPredictor(object):
    """
    Method: Neighbourhood Adaptive Prediction Sets
    Paper: Distribution Free Prediction Sets for Node Classification (Clarkson et al., 2023)
    Link: https://proceedings.mlr.press/v202/clarkson23a/clarkson23a.pdf
    Github: https://github.com/jase-clarkson/graph_cp/tree/master

    :param G: network of test graph data.
    :param cutoff: nodes with at least 'cutoff' k-hop neighbors for test.
    :param k: k-hop neighbors.
    :param scheme: name of decay rate.
    """

    def __init__(self, G, cutoff=50, k=2, scheme="unif"):
        super().__init__()

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
        quantiles_nb = []
        for node in list(self._G.nodes):
            t = self.calibrate_nbhd(node, probs, labels, alpha)
            quantiles_nb.append(t)
        nz = [p for p in quantiles_nb if p is not None]
        res = {}
        for p in nz:
            res.update(p)
        nbhd_quantiles = pd.Series(res, name='quantile')
        lcc_nodes = torch.tensor(nbhd_quantiles.index.values, device=self._device)
        quantiles = torch.tensor(nbhd_quantiles.values[:, None], device=self._device)
        prediction_sets = self.predict(probs[lcc_nodes], quantiles)
        prediction_sets = [tensor.cpu().tolist() for tensor in prediction_sets]
        return lcc_nodes, prediction_sets

    def calibrate_nbhd(self, node, probs, labels, alpha):
        node_ids, weights = self.get_nbhd_weights(node)

        if self._cutoff <= node_ids.shape[0]:
            quantile = self.calibrate_weighted(probs[node_ids], labels[node_ids],
                                               weights, alpha)
            return {node: quantile}
        return None

    def get_nbhd_weights(self, node):
        # Get dict containing nodes -> shortest path to node (i.e. depth).
        node_depth_map = pd.Series(nx.single_source_shortest_path_length(self._G, node, cutoff=self._k), name='distance')
        node_depth_map.index.name = 'node_id'
        node_depth_map = node_depth_map.drop(node) # Remove the node itself from list.
        node_depth_map = node_depth_map.reset_index()

        if self._scheme == 'unif':
            node_depth_map['weight'] = 1
        elif self._scheme == 'linear':
            node_depth_map['weight'] = 1 / node_depth_map['distance']
        elif self._scheme == 'geom':
            node_depth_map['weight'] = (0.5)**(node_depth_map['distance'] - 1)
        
        node_ids = torch.tensor(node_depth_map['node_id']).to(self._device)
        weights = torch.tensor(node_depth_map['weight']).to(self._device)

        return node_ids, weights
        
    def calibrate_weighted(self, probs, labels, weights, alpha):
        n = probs.shape[0]
        if n == 0:
            return alpha
        # Calibrate
        calibrator = ProbabilityAccumulator(probs)
        eps = torch.rand(n).to(self._device)
        alpha_max = calibrator.calibrate_scores(labels, eps)
        scores = alpha - alpha_max
        alpha_correction = self.get_weighted_quantile(scores, weights, alpha)
        return alpha - alpha_correction
    
    def get_weighted_quantile(self, scores, weights, alpha):
        wtildes = weights / (weights.sum() + 1)
        def critical_point_quantile(q): return (wtildes * (scores <= q)).sum().item() - (1 - alpha)
        try:
            q = brentq(critical_point_quantile, -1000, 1000)
        except ValueError:
            print(critical_point_quantile(-1000), critical_point_quantile(1000))
            print(scores)
            q = 0
        return q
    
    def predict(self, probs, alpha, allow_empty=True):
        n = probs.shape[0]
        eps = torch.rand(n, device=self._device)
        predictor = ProbabilityAccumulator(probs)
        S_hat = predictor.predict_sets(alpha, eps, allow_empty)
        return S_hat