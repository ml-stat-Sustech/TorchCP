import pytest
import math
import warnings

import torch
from torch.utils.data import Dataset

from torchcp.classification.scores import THR
from torchcp.classification.utils.metrics import Metrics
from torchcp.classification.predictors import ClassWisePredictor


@pytest.fixture
def mock_dataset():
    class MyDataset(Dataset):
        def __init__(self):
            self.x = torch.randn(100, 3)
            self.labels = torch.randint(0, 3, (100, ))

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
    return ClassWisePredictor(mock_score_function, mock_model)


def test_valid_initialization(predictor, mock_score_function, mock_model):
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model
    assert not predictor._model.training
    assert predictor._device == next(mock_model.parameters()).device
    assert predictor._logits_transformation.temperature == 1.0
    assert predictor.q_hat is None


@pytest.mark.parametrize("temperature", [-0.1, -1.0])
def test_invalid_initialization(mock_score_function, mock_model, temperature):
    with pytest.raises(ValueError, match="temperature must be greater than 0."):
        ClassWisePredictor(mock_score_function, mock_model, temperature)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 2])
def test_invalid_calibrate_alpha(predictor, alpha):
    logits = torch.randn(100, 3)
    labels = torch.randint(0, 3, (100, ))
    with pytest.raises(ValueError, match="alpha should be a value"):
        predictor.calculate_threshold(logits, labels, alpha)


@pytest.mark.parametrize("alpha", [0.05])
def test_calculate_threshold(predictor, mock_score_function, alpha):
    logits = torch.randn(100, 3)
    zeros = torch.zeros(30, dtype=torch.long)
    ones = torch.ones(30, dtype=torch.long)
    twos = torch.full((40,), 2, dtype=torch.long)
    labels = torch.cat([zeros, ones, twos])

    predictor.calculate_threshold(logits, labels, alpha)

    num_classes = 3
    scores = mock_score_function(logits, labels)
    excepted_qhat = torch.zeros(num_classes)
    for label in range(num_classes):
        temp_scores = scores[labels == label]
        excepted_qhat[label] = torch.sort(temp_scores).values[math.ceil((temp_scores.shape[0] + 1) * (1 - alpha)) - 1]
    assert torch.equal(predictor.q_hat, excepted_qhat)

    logits = torch.randn(100, 3)
    labels = torch.cat([torch.zeros((90, )), torch.ones((10, ))]).long()
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        predictor.calculate_threshold(logits, labels, alpha)
    assert len(warns) == 2
    assert "The number of scores is 0" in str(warns[1].message)
    assert "The value of quantile exceeds 1" in str(warns[0].message)

    scores = mock_score_function(logits, labels)
    excepted_qhat = torch.zeros(num_classes)
    marginal_q_hat = torch.sort(scores).values[math.ceil((scores.shape[0] + 1) * (1 - alpha)) - 1]
    for label in range(num_classes):
        temp_scores = scores[labels == label]
        if len(temp_scores) == 0:
            excepted_qhat[label] = marginal_q_hat
            continue
        quantile_value = math.ceil(temp_scores.shape[0] + 1) * (1 - alpha) / temp_scores.shape[0]
        if quantile_value > 1:
            excepted_qhat[label] = marginal_q_hat
            continue
        excepted_qhat[label] = torch.sort(temp_scores).values[math.ceil((temp_scores.shape[0] + 1) * (1 - alpha)) - 1]
    assert torch.equal(predictor.q_hat, excepted_qhat)