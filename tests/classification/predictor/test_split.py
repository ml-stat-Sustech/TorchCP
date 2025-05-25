# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import pytest
import torch
from torch.utils.data import Dataset

from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.predictor.base import BasePredictor
from torchcp.classification.score import THR
from torchcp.classification.utils.metrics import Metrics


@pytest.fixture
def mock_dataset():
    class MyDataset(Dataset):
        def __init__(self):
            self.x = torch.randn(100, 3)
            self.labels = torch.randint(0, 3, (100,))

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.labels[idx]

    return MyDataset()


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x

    return MockModel()


@pytest.fixture
def mock_score_function():
    return THR(score_type="softmax")


@pytest.fixture
def predictor(mock_score_function, mock_model):
    return SplitPredictor(mock_score_function, mock_model)


def test_base_predictor_abstractmethod(mock_score_function):
    class Predictor(BasePredictor):
        def __init__(self, score_function):
            super().__init__(score_function)

    predictor = Predictor(mock_score_function)
    with pytest.raises(NotImplementedError):
        predictor.calibrate(None, 0.1)
    with pytest.raises(NotImplementedError):
        predictor.predict(None)


def test_valid_initialization(predictor, mock_score_function, mock_model):
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model
    assert not predictor._model.training
    assert predictor._device == next(mock_model.parameters()).device
    assert predictor._logits_transformation.temperature == 1.0


def test_initialization_device(mock_score_function, mock_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = SplitPredictor(mock_score_function, mock_model, device=device)
    assert predictor._device == device
    

@pytest.mark.parametrize("temperature", [-0.1, -1.0])
def test_invalid_initialization(mock_score_function, mock_model, temperature):
    with pytest.raises(ValueError, match="temperature must be greater than 0."):
        SplitPredictor(mock_score_function, mock_model, temperature)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calibrate(predictor, mock_dataset, mock_score_function, mock_model, alpha):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)

    predictor.calibrate(cal_dataloader, alpha)

    mock_model.eval()
    logits = mock_model(mock_dataset.x)
    scores = mock_score_function(logits, mock_dataset.labels)
    excepted_qhat = torch.sort(scores).values[math.ceil((scores.shape[0] + 1) * (1 - alpha)) - 1]
    assert predictor.q_hat == excepted_qhat


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 2])
def test_invalid_calibrate_alpha(predictor, mock_dataset, alpha):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    with pytest.raises(ValueError, match="alpha should be a value"):
        predictor.calibrate(cal_dataloader, alpha)


def test_invalid_calibrate_model(mock_score_function, mock_dataset):
    predictor = SplitPredictor(mock_score_function, None)
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    with pytest.raises(ValueError, match="Model is not defined"):
        predictor.calibrate(cal_dataloader, 0.1)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calculate_threshold(predictor, mock_score_function, alpha):
    logits = torch.randn(100, 3)
    labels = torch.randint(0, 3, (100,))

    predictor.calculate_threshold(logits, labels, alpha)

    scores = mock_score_function(logits, labels)
    excepted_qhat = torch.sort(scores).values[math.ceil((scores.shape[0] + 1) * (1 - alpha)) - 1]
    assert predictor.q_hat == excepted_qhat


@pytest.mark.parametrize("q_hat", [0.5, 0.7])
def test_predict(predictor, mock_score_function, mock_model, mock_dataset, q_hat):
    predictor.q_hat = q_hat
    pred_sets = predictor.predict(mock_dataset.x)

    logits = mock_model(mock_dataset.x)
    scores = mock_score_function(logits)
    excepted_sets = (scores <= q_hat).int()
    assert torch.equal(pred_sets, excepted_sets)


def test_invalid_predict_model(mock_score_function, mock_dataset):
    predictor = SplitPredictor(mock_score_function, None)
    with pytest.raises(ValueError, match="Model is not defined"):
        predictor.predict(mock_dataset.x)


def test_q_hat_value_error(mock_model, mock_score_function):
    """Test that a ValueError is raised when self.q_hat is None and q_hat is not provided."""
    predictor = SplitPredictor(score_function=mock_score_function, model=mock_model)

    # Ensure self.q_hat is None
    predictor.q_hat = None

    logits = torch.rand(10, 5)

    with pytest.raises(ValueError, match="Ensure self.q_hat is not None. Please perform calibration first."):
        predictor.predict_with_logits(logits)


@pytest.mark.parametrize("q_hat", [0.5, 0.7])
def test_evaluate(predictor, mock_score_function, mock_model, mock_dataset, q_hat):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.q_hat = q_hat
    results = predictor.evaluate(cal_dataloader)

    logits = mock_model(mock_dataset.x)
    scores = mock_score_function(logits)
    excepted_sets = (scores <= q_hat).int()

    metrics = Metrics()
    assert len(results) == 2
    assert results['coverage_rate'] == metrics('coverage_rate')(excepted_sets, mock_dataset.labels)
    assert results['average_size'] == metrics('average_size')(excepted_sets)
