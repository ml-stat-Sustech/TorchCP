import pytest

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from torchcp.classification.scores import THR
from torchcp.graph.predictors import GraphSplitPredictor


@pytest.fixture
def mock_graph_data():
    num_nodes = 5
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0]
    ])
    x = torch.randn(num_nodes, 3)
    y = torch.tensor([0, 1, 2, 0, 1])
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def forward(self, x, edge_index):
            return torch.matmul(x, torch.randn(x.size(1), 3))
    return MockModel()


@pytest.fixture
def mock_score_function():
    def score_function(logits):
        return logits.max(dim=1).values
    return score_function


@pytest.fixture
def predictor(mock_graph_data, mock_score_function, mock_model):
    return GraphSplitPredictor(mock_graph_data, mock_score_function, mock_model)


def test_initialization(predictor, mock_graph_data, mock_score_function, mock_model):
    assert predictor._graph_data is mock_graph_data
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model

def test_calculate_threshold(predictor):
    logits = torch.randn(20, 3)
    cal_idx = torch.arange(19)
    labels = torch.randint(0, 3, (19, ))
    label_mask = F.one_hot(labels).bool()
    alpha = 0.1

    predictor.calculate_threshold(logits, cal_idx, label_mask, alpha)
    assert hasattr(predictor, 'q_hat')
    assert predictor.q_hat is not None