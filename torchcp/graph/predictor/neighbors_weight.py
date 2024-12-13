# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import networkx as nx
import torch
from scipy.optimize import brentq
from torch_geometric.utils.convert import to_networkx

from torchcp.classification.score import APS
from .split import SplitPredictor

DEFAULT_SCHEMES = ["unif", "linear", "geom"]


class NAPSPredictor(SplitPredictor):
    """
    Method: Neighbourhood Adaptive Prediction Sets
    Paper: Distribution Free Prediction Sets for Node Classification (Clarkson et al., 2023)
    Link: https://proceedings.mlr.press/v202/clarkson23a/clarkson23a.pdf
    Github: https://github.com/jase-clarkson/graph_cp/tree/master

    This class implements the NAPS method for conformal prediction on graph-structured data.
    It constructs prediction sets for nodes based on their neighborhood structure and non-conformity scores.

    Args:
        score_function (callable): 
            Must be APS non-conformity scores function with score_type="softmax"

        cutoff (int): Minimum number of k-hop neighbors a node must have to be included in the test set.
            Default is 50. Nodes with fewer than this number of neighbors will be excluded.

        k (int): Number of k-hop neighbors to include in the calibration set for each node.
            Default is 2, meaning nodes and their 2-hop neighbors are used for calibration.

        scheme (str): The weight decay scheme for k-hop neighbors. Options include:
            - 'unif': Uniform weighting (weights = 1)
            - 'linear': Linear decay (weights = 1/k)
            - 'geom': Geometric decay (weights = 2^{-(k-1)})
            Default is 'unif'.
    """

    def __init__(self, graph_data, score_function=APS(score_type="softmax"), model=None, cutoff=50, k=2, scheme="unif"):
        if type(score_function) is not APS:
            raise ValueError(
                f"Invalid score_function: {type(score_function).__name__}. Must be APS.")
        if score_function.score_type != "softmax":
            raise ValueError(
                f"Invalid score_type of APS: {score_function.score_type}. Must be softmax.")

        super().__init__(graph_data, score_function, model)

        if scheme not in DEFAULT_SCHEMES:
            raise ValueError(
                f"Invalid scheme: {scheme}. Choose from {DEFAULT_SCHEMES}")

        test_subgraph = self._graph_data.subgraph(self._graph_data.test_mask)
        self._G = to_networkx(test_subgraph).to_undirected()
        self._cutoff = cutoff
        self._k = k
        self._scheme = scheme

    # The calibration process ########################################################

    def calculate_threshold_for_node(self, node, logits, labels, alpha):
        """
        Calculate the conformal prediction threshold for a given node based on its neighborhood.

        This method computes the conformal prediction threshold for a specific node by examining
        the non-conformity scores of the node's neighbors. If the node has enough neighbors (as defined
        by the cutoff), it calibrates the threshold using these neighbors' scores.

        Args:
            node (int): 
                The ID of the node for which the threshold is being calculated.

            logits (torch.Tensor): 
                The raw model outputs (logits) for test nodes. Shape: [num_test_nodes, num_classes].

            labels (torch.Tensor): 
                The true labels for test nodes. Shape: [num_test_nodes].

            alpha (float): 
                The significance level for the conformal prediction. This is used to determine the 
                threshold for the prediction set.

        Returns:
            dict: 
                A dictionary where the key is the node ID and the value is the calibrated threshold 
                for the node. If the node doesn't have enough neighbors (i.e., fewer than `cutoff`), 
                `None` is returned.
        """
        node_ids, weights = self._get_nbhd_weights(node)

        if self._cutoff <= node_ids.shape[0]:
            quantile = self._calibrate_quantile(
                logits[node_ids], labels[node_ids], weights, alpha)
            return {node: quantile}
        return None

    def _get_nbhd_weights(self, node):
        """
        Get the neighboring nodes and their corresponding weights based on the chosen weight decay scheme.

        This method computes the neighbors of a given node within `k`-hop distance and assigns a weight
        to each neighbor according to the specified decay scheme.

        Args:
            node (int): The ID of the node for which neighbors and weights are being calculated.

        Returns:
            tuple: A tuple containing:
                - node_ids (torch.Tensor): A tensor of node IDs representing the neighbors of the given node.
                - weights (torch.Tensor): A tensor of weights corresponding to the neighbors. The weights are 
                calculated based on the decay scheme chosen during initialization.
        """
        # Get dict containing nodes -> shortest path to node (i.e. depth).
        neigh_depth = nx.single_source_shortest_path_length(
            self._G, node, cutoff=self._k)
        neigh_depth.pop(node, None)  # Remove the node itself from list.
        neigh_count = len(neigh_depth)

        node_ids = torch.tensor(list(neigh_depth.keys()), device=self._device)
        if self._scheme == 'unif':
            weights = torch.ones((neigh_count,), device=self._device)
        elif self._scheme == 'linear':
            weights = 1. / \
                      torch.tensor(list(neigh_depth.values()), device=self._device)
        elif self._scheme == 'geom':
            weights = (
                          0.5) ** (torch.tensor(list(neigh_depth.values()), device=self._device) - 1)

        return node_ids, weights

    def _calibrate_quantile(self, logits, labels, weights, alpha):
        """
        Calibrate the conformal prediction threshold by computing the quantile of non-conformity scores.

        This method calculates the correction to the significance level `alpha` by adjusting it based on
        the non-conformity scores of the calibration set, weighted by the provided `weights`. The correction
        is computed by finding the quantile of the scores, which is used to refine the threshold for prediction.

        Args:
            logits (torch.Tensor): 
                The raw model outputs (logits) for all nodes in the calibration set. Shape: [num_samples, num_classes].

            labels (torch.Tensor): 
                The true labels for all nodes in the calibration set. Shape: [num_samples].

            weights (torch.Tensor): 
                The weights corresponding to each sample in the calibration set. Shape: [num_samples].

            alpha (float): 
                The desired significance level for the conformal prediction.

        Returns:
            float: 
                The adjusted significance level (alpha) after calibration. This value is used to refine the conformal 
                prediction threshold based on the non-conformity scores.
        """
        if logits.shape[0] == 0:
            return alpha

        alpha_max = 1 - self.score_function(logits, labels)
        scores = alpha - alpha_max
        alpha_correction = self._get_weighted_quantile(scores, weights, alpha)
        return alpha - alpha_correction

    def _get_weighted_quantile(self, scores, weights, alpha):
        """
        Calculate the weighted quantile of the non-conformity scores.

        This method calculates the quantile `q` such that the weighted sum of the samples with scores less than or 
        equal to `q` is equal to the desired significance level `(1 - alpha)`. The quantile is computed using the 
        Brent method to find the root of the critical point function.

        Parameters:
            scores (torch.Tensor): 
                A tensor containing the non-conformity scores of the samples. Shape: [num_samples].

            weights (torch.Tensor): 
                A tensor containing the weights corresponding to each sample. Shape: [num_samples].

            alpha (float): 
                The desired significance level for the conformal prediction, typically a small positive value (e.g., 0.05).

        Returns:
            float: 
                The quantile `q` such that the weighted sum of the samples with scores less than or equal to `q` 
                is equal to the desired significance level `(1 - alpha)`.
        """
        wtildes = weights / (weights.sum() + 1)

        def critical_point_quantile(q):
            return (
                    wtildes * (scores <= q)).sum().item() - (1 - alpha)

        try:
            q = brentq(critical_point_quantile, -1000, 1000)
        except ValueError:
            q = 0
            raise ValueError(
                "Did not find a suitable alpha value, keeping alpha unchanged.")
        return q

    # The prediction process ########################################################

    def precompute_naps_sets(self, logits, labels, alpha):
        """
        Precompute the prediction sets for nodes in the graph based on logits and labels.

        This method calculates the prediction sets for each node that has at least 'cutoff' k-hop neighbors, 
        based on the provided logits and labels. The prediction sets are precomputed for a given empirical 
        marginal coverage `1 - alpha`, where alpha is the significance level.

        Parameters:
            logits (torch.Tensor):
                A tensor containing the model's predicted logits for each test node. Shape: [num_test_nodes, num_classes].

            labels (torch.Tensor):
                A tensor containing the true labels of test nodes. Shape: [num_test_nodes].

            alpha (float):
                The pre-defined empirical marginal coverage level, where `1 - alpha` represents the confidence 
                level of the prediction sets.

        Returns:
            lcc_nodes (torch.Tensor):
                A tensor containing the indices of the nodes that meet the criteria of having at least 'cutoff' k-hop 
                neighbors for testing. Shape: [num_lcc_nodes].

            prediction_sets (list):
                A list containing the precomputed prediction sets for each node in `lcc_nodes`. Each set is a 
                list of predicted classes for that node.
        """
        quantiles_nb = {}
        for node in list(self._G.nodes):
            p = self.calculate_threshold_for_node(node, logits, labels, alpha)
            if p is not None:
                quantiles_nb.update(p)

        lcc_nodes = torch.tensor(
            list(quantiles_nb.keys()), device=self._device)
        quantiles = torch.tensor(
            list(quantiles_nb.values()), device=self._device)
        prediction_sets = self.predict(logits[lcc_nodes], quantiles[:, None])
        return lcc_nodes, prediction_sets

    def predict(self, logits, alphas):
        """
        Generate prediction sets for each node based on the logits and the corresponding adjusted alphas.

        This method uses the scores derived from the logits to create prediction sets for each sample
        (e.g., nodes in a graph). The size of each prediction set is determined by the corresponding `alpha` value.

        Args:
            logits (torch.Tensor):
                A tensor of model outputs (logits), where each row corresponds to the logits for one node/sample.
                Shape: [num_samples, num_classes].

            alphas (torch.Tensor):
                A tensor containing the significance levels (alphas) for each sample. Each alpha corresponds to 
                the desired size of the prediction set for the respective sample.
                Shape: [num_samples].

        Returns:
            List:
                A list of prediction sets, one for each sample. Each set contains the nodes/classes that are 
                predicted to belong to the true class for the given significance level.
        """
        scores = self.score_function(logits)
        prediction_set_list = []
        for index in range(scores.shape[0]):
            prediction_set_list.append(self._generate_prediction_set(
                scores[index, :].reshape(1, -1), 1 - alphas[index]))

        return torch.cat(prediction_set_list, dim=0)
