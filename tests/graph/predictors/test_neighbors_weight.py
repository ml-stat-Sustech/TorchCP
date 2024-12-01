import pytest
import torch
import networkx as nx

from torchcp.classification.scores import THR
from torchcp.graph.predictors.neighbors_weight import NAPSPredictor

@pytest.fixture
def mock_graph_data():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    G.graph['test_mask'] = [True, True, True, False, False]
    return G

@pytest.fixture
def mock_logits():
    return torch.tensor([[0.8, 0.2],
                         [0.6, 0.4],
                         [0.1, 0.9],
                         [0.7, 0.3],
                         [0.3, 0.7]])

@pytest.fixture
def mock_labels():
    return torch.tensor([0, 1, 1, 0, 1])

@pytest.fixture
def naps_predictor(mock_graph_data):
    return NAPSPredictor(graph_data=mock_graph_data, score_function=THR(score_type="softmax"), cutoff=1, k=1, scheme="unif")

def test_init_valid_naps_predictor(mock_graph_data):
    predictor = NAPSPredictor(graph_data=mock_graph_data, score_function=THR(score_type="softmax"), cutoff=50, k=2, scheme="unif")
    assert predictor._cutoff == 50
    assert predictor._k == 2
    assert predictor._scheme == "unif"

def test_init_invalid_score_function(mock_graph_data):
    with pytest.raises(ValueError, match="Invalid score_function"):
        NAPSPredictor(graph_data=mock_graph_data, score_function="invalid_score_function")

def test_calculate_threshold_for_node(mock_logits, mock_labels, naps_predictor):
    threshold = naps_predictor.calculate_threshold_for_node(0, mock_logits, mock_labels, alpha=0.05)
    assert isinstance(threshold, dict)
    assert 0 in threshold or threshold is None

def test_precompute_naps_sets(mock_logits, mock_labels, naps_predictor):
    lcc_nodes, prediction_sets = naps_predictor.precompute_naps_sets(mock_logits, mock_labels, alpha=0.05)
    assert isinstance(lcc_nodes, torch.Tensor)
    assert isinstance(prediction_sets, list)
    assert len(prediction_sets) == len(lcc_nodes)

def test_predict(mock_logits, naps_predictor):
    alphas = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    prediction_sets = naps_predictor.predict(mock_logits, alphas)
    assert isinstance(prediction_sets, list)
    assert len(prediction_sets) == len(mock_logits)

def test_invalid_scheme(mock_graph_data):
    with pytest.raises(ValueError, match="Invalid scheme"):
        NAPSPredictor(graph_data=mock_graph_data, score_function=THR(score_type="softmax"), scheme="invalid_scheme")

def test_get_nbhd_weights(naps_predictor):
    node_ids, weights = naps_predictor._get_nbhd_weights(0)
    assert isinstance(node_ids, torch.Tensor)
    assert isinstance(weights, torch.Tensor)

def test_calibrate_quantile(mock_logits, mock_labels, naps_predictor):
    weights = torch.ones_like(mock_labels, dtype=torch.float)
    quantile = naps_predictor._calibrate_quantile(mock_logits, mock_labels, weights, alpha=0.05)
    assert isinstance(quantile, float)

def test_get_weighted_quantile(mock_logits, naps_predictor):
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    weights = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)
    quantile = naps_predictor._get_weighted_quantile(scores, weights, alpha=0.05)
    assert isinstance(quantile, float)
