import pytest

import math
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from torchcp.classification.scores import THR
from torchcp.graph.predictors import GraphSplitPredictor


@pytest.fixture
def mock_graph_data():
    num_nodes = 200
    x = torch.randn(num_nodes, 3)

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0]
    ])
    y = torch.randint(0, 3, (num_nodes, ))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, edge_indes):
            return x
    return MockModel()


@pytest.fixture
def mock_score_function():
    return THR(score_type="softmax")


@pytest.fixture
def predictor(mock_graph_data, mock_score_function, mock_model):
    return GraphSplitPredictor(mock_graph_data, mock_score_function, mock_model)


def test_initialization(predictor, mock_graph_data, mock_score_function, mock_model):
    assert predictor._graph_data is mock_graph_data
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calculate(predictor, mock_graph_data, mock_score_function, alpha):
    num_nodes = mock_graph_data.x.shape[0]
    num_calib = int(num_nodes / 2)

    label_mask = F.one_hot(mock_graph_data.y).bool()
    cal_idx = torch.arange(num_calib)

    predictor.calibrate(cal_idx, alpha)

    scores = mock_score_function(mock_graph_data.x)
    cal_scores = scores[cal_idx][label_mask[cal_idx]]
    quantile = torch.sort(cal_scores).values[math.ceil((cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    assert predictor.q_hat == quantile


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calculate_threshold(predictor, mock_graph_data, mock_score_function, alpha):
    num_nodes = mock_graph_data.x.shape[0]
    num_calib = int(num_nodes / 2)

    logits = torch.randn(num_nodes, 3)
    label_mask = F.one_hot(mock_graph_data.y).bool()
    cal_idx = torch.arange(num_calib)

    predictor.calculate_threshold(logits, cal_idx, label_mask, alpha)

    scores = mock_score_function(logits)
    cal_scores = scores[cal_idx][label_mask[cal_idx]]
    quantile = torch.sort(cal_scores).values[math.ceil((cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    assert predictor.q_hat == quantile

@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_predict(predictor, mock_graph_data, mock_score_function, alpha):
    num_nodes = mock_graph_data.x.shape[0]
    num_calib = int(num_nodes / 2)

    cal_idx = torch.arange(num_nodes)[:num_calib]
    eval_idx = torch.arange(num_nodes)[num_calib:]

    with pytest.raises(ValueError, match="Ensure self.q_hat is not None. Please perform calibration first."):
        predictor.predict(eval_idx)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_predict_with_logits(predictor, mock_graph_data, mock_score_function, alpha):
    num_nodes = mock_graph_data.x.shape[0]
    num_calib = int(num_nodes / 2)

    logits = torch.randn(num_nodes, 3)
    label_mask = F.one_hot(mock_graph_data.y).bool()
    cal_idx = torch.arange(num_nodes)[:num_calib]
    eval_idx = torch.arange(num_nodes)[num_calib:]

    scores = mock_score_function(logits)
    cal_scores = scores[cal_idx][label_mask[cal_idx]]
    quantile = torch.sort(cal_scores).values[math.ceil((cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    eval_scores = scores[eval_idx]
    excepted_sets = [torch.argwhere(eval_scores[i] <= quantile).reshape(-1).tolist() for i in range(eval_scores.shape[0])]

    pred_sets = predictor.predict_with_logits(logits, eval_idx, quantile)
    assert excepted_sets == pred_sets

    with pytest.raises(ValueError, match="Ensure self.q_hat is not None. Please perform calibration first."):
        predictor.predict_with_logits(logits, eval_idx)

# def test_evaluate(predictor):
#     eval_idx = torch.tensor([3, 4])
#     metrics = predictor.evaluate(eval_idx)
    
#     assert isinstance(metrics, dict)
#     assert "Coverage_rate" in metrics
#     assert "Average_size" in metrics
#     assert "Singleton_Hit_Ratio" in metrics
#     assert all(isinstance(value, float) for value in metrics.values())

# def test_invalid_calibration(predictor):
#     eval_idx = torch.tensor([3, 4])
#     with pytest.raises(AssertionError, match="Ensure self.q_hat is not None. Please perform calibration first."):
#         predictor.predict_with_logits(torch.randn(5, 3), eval_idx, q_hat=None)