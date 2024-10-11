# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial
from scipy.optimize import brentq
from tqdm.contrib.concurrent import process_map


from .base import BaseGraphPredictor
from .utils import ProbabilityAccumulator


class NAPSSplitPredictor(BaseGraphPredictor):
    """
    Neighbourhood Adaptive Prediction Sets (Clarkson et al., 2023)
    paper: https://proceedings.mlr.press/v202/clarkson23a/clarkson23a.pdf
    github: https://github.com/jase-clarkson/graph_cp/tree/master

    :param score_function: basic non-conformity score function.
    :param model: a pytorch model.
    """

    def __init__(self, score_function, k=2, scheme="unif", model=None):
        super().__init__(score_function, model)

        if scheme not in ["unif", "linear", "geom"]:
            raise NotImplementedError

        self._k = k
        self._scheme = "unif"

    def get_nbhd_weights(G, node, k=2, scheme='unif'):
        # Get dict containing nodes -> shortest path to node (i.e. depth).
        node_depth_map = pd.Series(nx.single_source_shortest_path_length(G, node, cutoff=k), name='distance')
        node_depth_map.index.name = 'node_id'
        node_depth_map = node_depth_map.drop(node) # Remove the node itself from list.
        node_depth_map = node_depth_map.reset_index()

        if scheme == 'geom':
            node_depth_map['weight'] = (0.5)**(node_depth_map['distance'] - 1)  # Weight =
        elif scheme == 'linear':
            node_depth_map['weight'] = 1 / node_depth_map['distance']
        else:
            node_depth_map['weight'] = 1
        return node_depth_map
    
    def calibrate_nbhd(G, node, logits, labels, alpha, cutoff = 50, scheme='unif'):
        nbs = self.get_nbhd_weights(G, node, k=2, scheme=scheme)
        nb_ids = nbs['node_id'].values
        weights = nbs['weight'].values
        if cutoff <= len(nb_ids):
            quantile = self.calibrate_weighted(logits.loc[nb_ids].values,
                                np.squeeze(labels.loc[nb_ids].values),
                                        weights, alpha)
            return {node: quantile}
        
    def calibrate_weighted(probs, labels, weights, alpha):
        n = probs.shape[0]
        if n == 0:
            return alpha
        # Calibrate
        calibrator = ProbabilityAccumulator(probs)
        eps = np.random.uniform(low=0, high=1, size=n)
        alpha_max = calibrator.calibrate_scores(labels, eps)
        scores = alpha - alpha_max
        alpha_correction = self.get_weighted_quantile(scores, weights, alpha)
        return alpha - alpha_correction
    
    def get_weighted_quantile(scores, weights, alpha):
        wtildes = weights / (weights.sum() + 1)
        def critical_point_quantile(q): return (wtildes * (scores <= q)).sum() - (1 - alpha)
        try:
            q = brentq(critical_point_quantile, -1000, 1000)
        except ValueError:
            print(critical_point_quantile(-1000), critical_point_quantile(1000))
            print(scores)
            q = 0
        return q
    
    def precompute_naps_sets(scheme):
        f = partial(self.calibrate_nbhd, scheme=scheme)
        quantiles_nb = process_map(f, list(G.nodes), max_workers=12)
        nz = [p for p in quantiles_nb if p is not None]
        res = {}
        for p in nz:
            res.update(p)
        nbhd_quantiles = pd.Series(res, name='quantile')
        nbhd_quantiles
        lcc_nodes = nbhd_quantiles.index.values
        sets_nb = self.predict(preds.loc[lcc_nodes].values, nbhd_quantiles.values[:, None])
        sets_nb = pd.Series(sets_nb, index=lcc_nodes, name='set')
        sets_nb = pd.DataFrame(sets_nb)
        sets_nb['set_size'] = sets_nb['set'].apply(len)
        sets_nb['covers'] = [test_y.loc[i].values in sets_nb.loc[i, 'set'] for i in sets_nb.index.values]
        return sets_nb, lcc_nodes
    
    def predict(probs, alpha, allow_empty=True):
        n = probs.shape[0]
        eps = np.random.uniform(0, 1, n)
        predictor = ProbabilityAccumulator(probs)
        S_hat = predictor.predict_sets(alpha, eps, allow_empty)
        return S_hat


