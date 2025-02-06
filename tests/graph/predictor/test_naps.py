# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random

import networkx as nx
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx

from torchcp.classification.score import APS, THR
from torchcp.graph.predictor.naps import NAPSPredictor


@pytest.fixture
def mock_graph_data():
    num_nodes = 200
    num_test = int(num_nodes * 0.9)
    num_edges = num_nodes * 20
    x = torch.randn(num_nodes, 3)

    edges = set()
    while len(edges) < num_edges:
        new_edge = sorted(random.sample(range(num_nodes), 2))
        if new_edge[0] != new_edge[1]:
            edges.add(tuple(new_edge))
            edges.add((new_edge[1], new_edge[0]))
    edge_index = torch.tensor(list(edges)).T.contiguous()

    y = torch.randint(0, 3, (num_nodes,))
    test_mask = torch.zeros(num_nodes).bool()
    test_mask[torch.randperm(num_nodes)[:num_test]] = True
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes, test_mask=test_mask)


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def inference(self, x, subgraph_loader=None):
            return x

    return MockModel()


@pytest.fixture
def naps_predictor(mock_graph_data):
    return NAPSPredictor(graph_data=mock_graph_data, score_function=APS(score_type="softmax"), cutoff=30, k=2,
                         scheme="unif")


def test_init_valid_naps_predictor(mock_graph_data):
    predictor = NAPSPredictor(graph_data=mock_graph_data, score_function=APS(score_type="softmax"), cutoff=50, k=2,
                              scheme="unif")

    test_subgraph = mock_graph_data.subgraph(mock_graph_data.test_mask)
    excepted_G = to_networkx(test_subgraph).to_undirected()
    assert nx.is_isomorphic(predictor._G, excepted_G)

    assert predictor._cutoff == 50
    assert predictor._k == 2
    assert predictor._scheme == "unif"


def test_init_invalid_naps_predictor(mock_graph_data):
    with pytest.raises(ValueError, match="Invalid score_function"):
        NAPSPredictor(graph_data=mock_graph_data, score_function=THR(score_type="softmax"))

    with pytest.raises(ValueError, match="Invalid score_type of APS"):
        NAPSPredictor(graph_data=mock_graph_data, score_function=APS(score_type="identity"))

    with pytest.raises(ValueError, match="Invalid scheme"):
        NAPSPredictor(graph_data=mock_graph_data, score_function=APS(score_type="softmax"), scheme="invalid_scheme")


def test_calculate_threshold_for_node(mock_graph_data):
    naps_predictor = NAPSPredictor(graph_data=mock_graph_data,
                                   score_function=APS(score_type="softmax"),
                                   cutoff=130, k=2, scheme="unif")
    logits = mock_graph_data.x[mock_graph_data.test_mask]
    labels = mock_graph_data.y[mock_graph_data.test_mask]
    for node_id in naps_predictor._G.nodes():
        neigh_depth = nx.single_source_shortest_path_length(naps_predictor._G, node_id, cutoff=2)
        if len(neigh_depth) >= 131:
            torch.manual_seed(42)
            node_alpha = naps_predictor.calculate_threshold_for_node(node_id, logits, labels, alpha=0.1)
            torch.manual_seed(42)
            node_ids, weights = naps_predictor._get_nbhd_weights(node_id)
            quantile = naps_predictor._calibrate_quantile(logits[node_ids], labels[node_ids], weights, alpha=0.1)
            assert node_id in node_alpha.keys()
            assert node_alpha[node_id] == quantile
        else:
            node_alpha = naps_predictor.calculate_threshold_for_node(node_id, logits, labels, alpha=0.1)
            assert node_alpha is None


@pytest.mark.parametrize("scheme", ["unif", "linear", "geom"])
def test_get_nbhd_weights(mock_graph_data, scheme):
    predictor = NAPSPredictor(graph_data=mock_graph_data, score_function=APS(score_type="softmax"), cutoff=50, k=2,
                              scheme=scheme)
    node_id = int(torch.where(mock_graph_data.test_mask)[0][0])

    node_ids, weights = predictor._get_nbhd_weights(node_id)

    neigh_depth = nx.single_source_shortest_path_length(predictor._G, node_id, cutoff=2)
    neigh_depth.pop(node_id, None)
    neigh_count = len(neigh_depth)

    assert torch.equal(node_ids, torch.tensor(list(neigh_depth.keys())))

    if scheme == "unif":
        assert torch.allclose(weights, torch.ones((neigh_count,)))
    elif scheme == 'linear':
        assert torch.allclose(weights, 1. / torch.tensor(list(neigh_depth.values())))
    elif scheme == 'geom':
        assert torch.allclose(weights, 0.5 ** (torch.tensor(list(neigh_depth.values())) - 1))


def test_calibrate_quantile(naps_predictor):
    logits = torch.tensor([])
    labels = torch.randint(0, 3, (100,))
    weights = torch.ones((100,), dtype=torch.float32)
    alpha = naps_predictor._calibrate_quantile(logits, labels, weights, alpha=0.05)
    assert alpha == 0.05

    logits = torch.randn(100, 3)
    torch.manual_seed(42)
    alpha = naps_predictor._calibrate_quantile(logits, labels, weights, alpha=0.05)

    torch.manual_seed(42)
    alpha_max = 1 - naps_predictor.score_function(logits, labels)
    scores = 0.05 - alpha_max
    alpha_correction = naps_predictor._get_weighted_quantile(scores, weights, alpha=0.05)
    assert alpha == (0.05 - alpha_correction)


def test_get_weighted_quantile(naps_predictor):
    scores = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    weights = torch.ones((9,), dtype=torch.float32)

    quantile = naps_predictor._get_weighted_quantile(scores, weights, alpha=0.5)
    assert quantile >= 0.3 and quantile < 0.35
    quantile = naps_predictor._get_weighted_quantile(scores, weights, alpha=0.44)
    assert quantile >= 0.3 and quantile < 0.35
    quantile = naps_predictor._get_weighted_quantile(scores, weights, alpha=0.66)
    assert quantile >= 0.2 and quantile < 0.25
    quantile = naps_predictor._get_weighted_quantile(scores, weights, alpha=0.23)
    assert quantile >= 0.4 and quantile < 0.45

    with pytest.raises(ValueError, match="Did not find a suitable alpha value"):
        naps_predictor._get_weighted_quantile(scores, weights, 1.5)


def test_predict(mock_graph_data, mock_model):
    naps_predictor = NAPSPredictor(graph_data=mock_graph_data,
                                   score_function=APS(score_type="softmax"),
                                   model=mock_model,
                                   cutoff=130, k=2, scheme="unif")

    eval_idx = torch.where(mock_graph_data.test_mask)[0]
    torch.manual_seed(42)

    logits = mock_graph_data.x[eval_idx]
    labels = mock_graph_data.y[eval_idx]
    quantiles_nb = {}
    for node_id in naps_predictor._G.nodes():
        neigh_depth = nx.single_source_shortest_path_length(naps_predictor._G, node_id, cutoff=2)
        if len(neigh_depth) >= 131:
            node_alpha = naps_predictor.calculate_threshold_for_node(node_id, logits, labels, alpha=0.1)
            quantiles_nb.update(node_alpha)
        else:
            naps_predictor.calculate_threshold_for_node(node_id, logits, labels, alpha=0.1)
    excepted_nodes = torch.tensor(list(quantiles_nb.keys()))
    quantiles = torch.tensor(list(quantiles_nb.values()))
    excepted_sets = naps_predictor._generate_prediction_set(logits[excepted_nodes], quantiles[:, None])

    torch.manual_seed(42)
    pred_nodes, pred_sets = naps_predictor.predict(eval_idx, alpha=0.1)

    assert torch.equal(excepted_nodes, pred_nodes)
    assert torch.equal(excepted_sets, pred_sets)


def test_predict_with_logits(mock_graph_data):
    naps_predictor = NAPSPredictor(graph_data=mock_graph_data,
                                   score_function=APS(score_type="softmax"),
                                   cutoff=130, k=2, scheme="unif")

    eval_idx = torch.where(mock_graph_data.test_mask)[0]
    torch.manual_seed(42)
    logits = mock_graph_data.x[eval_idx]
    labels = mock_graph_data.y[eval_idx]
    quantiles_nb = {}
    for node_id in naps_predictor._G.nodes():
        neigh_depth = nx.single_source_shortest_path_length(naps_predictor._G, node_id, cutoff=2)
        if len(neigh_depth) >= 131:
            node_alpha = naps_predictor.calculate_threshold_for_node(node_id, logits, labels, alpha=0.1)
            quantiles_nb.update(node_alpha)
        else:
            naps_predictor.calculate_threshold_for_node(node_id, logits, labels, alpha=0.1)
    excepted_nodes = torch.tensor(list(quantiles_nb.keys()))
    quantiles = torch.tensor(list(quantiles_nb.values()))
    excepted_sets = naps_predictor._generate_prediction_set(logits[excepted_nodes], quantiles[:, None])

    torch.manual_seed(42)
    pred_nodes, pred_sets = naps_predictor.predict_with_logits(mock_graph_data.x, eval_idx, alpha=0.1)

    assert torch.equal(excepted_nodes, pred_nodes)
    assert torch.equal(excepted_sets, pred_sets)


def test_evaluate(mock_graph_data, mock_model):

    naps_predictor = NAPSPredictor(graph_data=mock_graph_data,
                                   score_function=APS(score_type="softmax"),
                                   model=mock_model,
                                   cutoff=130, k=2, scheme="unif")
    eval_idx = torch.where(mock_graph_data.test_mask)[0]

    torch.manual_seed(0)
    results = naps_predictor.evaluate(eval_idx, alpha=0.1)

    torch.manual_seed(0)
    lcc_nodes, prediction_sets = naps_predictor.predict_with_logits(mock_graph_data.x, eval_idx, 0.1)
    labels = mock_graph_data.y[eval_idx][lcc_nodes]

    except_resultsres_dict = {"coverage_rate": naps_predictor._metric('coverage_rate')(prediction_sets, labels),
                    "average_size": naps_predictor._metric('average_size')(prediction_sets, labels),
                    "singleton_hit_ratio": naps_predictor._metric('singleton_hit_ratio')(prediction_sets, labels)}
    
    assert results["coverage_rate"] == except_resultsres_dict["coverage_rate"]
    assert results["average_size"] == except_resultsres_dict["average_size"]
    assert results["singleton_hit_ratio"] == except_resultsres_dict["singleton_hit_ratio"]
