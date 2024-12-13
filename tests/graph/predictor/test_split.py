# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from torchcp.classification.score import THR
from torchcp.graph.predictor import SplitPredictor
from torchcp.graph.predictor.base import BasePredictor
from torchcp.graph.utils import Metrics


@pytest.fixture
def mock_graph_data():
    num_nodes = 200
    x = torch.randn(num_nodes, 3)

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0]
    ])
    y = torch.randint(0, 3, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, edge_index=None):
            return x

    return MockModel()


@pytest.fixture
def mock_score_function():
    return THR(score_type="softmax")


@pytest.fixture
def predictor(mock_graph_data, mock_score_function, mock_model):
    return SplitPredictor(mock_graph_data, mock_score_function, mock_model)


@pytest.fixture
def preprocess(mock_graph_data, mock_score_function, mock_model):
    class PreProcess(object):
        def __init__(self):
            num_nodes = mock_graph_data.x.shape[0]
            num_calib = int(num_nodes / 2)

            self.label_mask = F.one_hot(mock_graph_data.y).bool()
            self.cal_idx = torch.arange(num_nodes)[:num_calib]
            self.eval_idx = torch.arange(num_nodes)[num_calib:]

            self.scores = mock_score_function(mock_model(mock_graph_data.x))
            self.cal_scores = self.scores[self.cal_idx][self.label_mask[self.cal_idx]]
            self.logits = mock_model(mock_graph_data.x)

    return PreProcess()


def test_base_graph_predictor(mock_graph_data, mock_score_function):
    class TestPredictor(BasePredictor):
        def __init__(self, graph_data, score_function, model=None):
            super().__init__(graph_data, score_function, model)

    predictor = TestPredictor(mock_graph_data, mock_score_function)
    with pytest.raises(NotImplementedError):
        predictor.calibrate(None, 0.1)

    with pytest.raises(NotImplementedError):
        predictor.predict(None)


def test_initialization(predictor, mock_graph_data, mock_score_function, mock_model):
    assert predictor._device == mock_graph_data.x.device
    assert predictor._graph_data is mock_graph_data
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calculate(predictor, preprocess, alpha):
    predictor.calibrate(preprocess.cal_idx, alpha)
    quantile = torch.sort(preprocess.cal_scores).values[
        math.ceil((preprocess.cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    assert predictor.q_hat == quantile


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calculate_threshold(predictor, preprocess, alpha):
    predictor.calculate_threshold(
        preprocess.logits, preprocess.cal_idx, preprocess.label_mask, alpha)
    quantile = torch.sort(preprocess.cal_scores).values[
        math.ceil((preprocess.cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    assert predictor.q_hat == quantile


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_predict(predictor, preprocess, alpha):
    with pytest.raises(ValueError, match="Ensure self.q_hat is not None. Please perform calibration first."):
        predictor.predict(preprocess.eval_idx)

    quantile = torch.sort(preprocess.cal_scores).values[
        math.ceil((preprocess.cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    eval_scores = preprocess.scores[preprocess.eval_idx]
    excepted_sets = (eval_scores <= quantile).int()

    predictor.calibrate(preprocess.cal_idx, alpha)
    pred_sets = predictor.predict(preprocess.eval_idx)
    assert torch.equal(excepted_sets, pred_sets)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_predict_with_logits(predictor, preprocess, alpha):
    quantile = torch.sort(preprocess.cal_scores).values[
        math.ceil((preprocess.cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    eval_scores = preprocess.scores[preprocess.eval_idx]
    excepted_sets = (eval_scores <= quantile).int()

    pred_sets = predictor.predict_with_logits(
        preprocess.logits, preprocess.eval_idx, quantile)
    assert torch.equal(excepted_sets, pred_sets)

    with pytest.raises(ValueError, match="Ensure self.q_hat is not None. Please perform calibration first."):
        predictor.predict_with_logits(preprocess.logits, preprocess.eval_idx)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_evaluate(predictor, mock_graph_data, preprocess, alpha):
    with pytest.raises(ValueError, match="Ensure self.q_hat is not None. Please perform calibration first."):
        predictor.evaluate(preprocess.eval_idx)

    quantile = torch.sort(preprocess.cal_scores).values[
        math.ceil((preprocess.cal_idx.shape[0] + 1) * (1 - alpha)) - 1]

    eval_scores = preprocess.scores[preprocess.eval_idx]
    excepted_sets = (eval_scores <= quantile).int()

    predictor.calibrate(preprocess.cal_idx, alpha)
    results = predictor.evaluate(preprocess.eval_idx)
    metrics = Metrics()
    assert len(results) == 3
    assert results['coverage_rate'] == metrics('coverage_rate')(
        excepted_sets, mock_graph_data.y[preprocess.eval_idx])
    assert results['average_size'] == metrics('average_size')(
        excepted_sets, mock_graph_data.y[preprocess.eval_idx])
    assert results['singleton_hit_ratio'] == metrics('singleton_hit_ratio')(
        excepted_sets, mock_graph_data.y[preprocess.eval_idx])
