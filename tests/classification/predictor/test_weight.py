# Copyright (c) 2023-present, SUSTech-ML and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchcp.classification.predictor.weight import WeightedPredictor
from torchcp.classification.score import APS as score_fn


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


class SimpleEncoder(nn.Module):
    """Simple image encoder for testing"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_score_function():
    """Score function fixture"""
    return score_fn()


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    n_samples = 100
    n_features = 10
    n_classes = 3

    # Create random data
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))

    # Create DataLoader
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)

    return dataloader


def test_weighted_predictor_init(simple_score_function):
    """Test WeightedPredictor initialization"""
    model = SimpleModel()
    encoder = SimpleEncoder()

    # Test successful initialization
    predictor = WeightedPredictor(
        score_function=simple_score_function,
        model=model,
        image_encoder=encoder
    )
    assert predictor.image_encoder is not None
    assert predictor.domain_classifier is None

    # Test initialization without image_encoder
    with pytest.raises(ValueError, match="image_encoder cannot be None"):
        WeightedPredictor(
            score_function=simple_score_function,
            model=model,
            image_encoder=None
        )


def test_calibrate(simple_score_function, sample_data):
    """Test calibration process"""
    model = SimpleModel()
    encoder = SimpleEncoder()
    predictor = WeightedPredictor(
        score_function=simple_score_function,
        model=model,
        image_encoder=encoder
    )
    predictor.calibrate(sample_data, None)

    # Test calibration with valid alpha
    predictor.calibrate(sample_data, alpha=0.1)
    assert predictor.alpha == 0.1
    assert predictor.scores is not None
    assert predictor.source_image_features is not None


def test_calculate_threshold(simple_score_function):
    """Test threshold calculation"""
    model = SimpleModel()
    encoder = SimpleEncoder()
    predictor = WeightedPredictor(
        score_function=simple_score_function,
        model=model,
        image_encoder=encoder
    )

    logits = torch.randn(10, 3)
    labels = torch.randint(0, 3, (10,))

    # Test valid alpha
    predictor.calculate_threshold(logits, labels, alpha=0.1)
    assert predictor.scores is not None
    assert len(predictor.scores) == len(labels) + 1
    assert torch.isinf(predictor.scores[-1])


def test_evaluate_without_calibration(simple_score_function, sample_data):
    """Test evaluate method without prior calibration"""
    model = SimpleModel()
    encoder = SimpleEncoder()
    predictor = WeightedPredictor(
        score_function=simple_score_function,
        model=model,
        image_encoder=encoder
    )

    with pytest.raises(ValueError, match="Please calibrate first"):
        predictor.evaluate(sample_data)


def test_predict_without_calibration():
    """Test predict without calibration"""
    predictor = WeightedPredictor(lambda x, y: x, image_encoder=SimpleEncoder())
    with pytest.raises(ValueError, match="Please calibrate first"):
        predictor.predict(torch.randn(10, 5))


def test_evaluate(simple_score_function, sample_data):
    """Test evaluate method with proper setup"""
    model = SimpleModel()
    encoder = SimpleEncoder()
    predictor = WeightedPredictor(
        score_function=simple_score_function,
        model=model,
        image_encoder=encoder
    )

    # Calibrate first
    predictor.calibrate(sample_data, alpha=0.1)

    # Test evaluation
    metrics = predictor.evaluate(sample_data)

    assert isinstance(metrics, dict)
    assert "coverage_rate" in metrics
    assert "average_size" in metrics
    assert 0 <= metrics["coverage_rate"] <= 1
    assert metrics["average_size"] >= 0


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("cal_scores, cal_weights", [
    ([0.1, 0.9], [9., 1.]),
    ([0.9, 0.1], [1., 9.]),
    ([0.1, 0.9], [1., 9.]),
    ([0.1, 0.9], [1., 1.]),
    ([0.1, 0.1], [9., 1.]),
    ([0.1, float("inf")], [9., 1.]),
    ([0.1] + [float("inf")] * 20, [100.] + [1.] * 20),
])
def test_weighted_quantile_score_weight_pairs(
    cal_scores, cal_weights, batch_size
):
    """Keep calibration weights paired with their own nonconformity scores."""

    class WeightFeature(nn.Module):
        def forward(self, x):
            return x[:, -1:]

    def scores(logits, labels=None):
        values = logits.reshape(-1, 4)[:, :3]
        if labels is None:
            return values
        return values[torch.arange(len(labels)), labels]

    X = torch.zeros(len(cal_scores), 4)
    X[:, 0] = torch.tensor(cal_scores)
    X[:, -1] = torch.tensor(cal_weights).log()
    y = torch.zeros(len(cal_scores), dtype=torch.long)
    classifier = nn.Sequential(nn.Linear(1, 2, bias=False), nn.Softmax(dim=1))
    with torch.no_grad():
        classifier[0].weight.copy_(torch.tensor([[0.], [1.]]))
    predictor = WeightedPredictor(
        scores, nn.Identity(), image_encoder=WeightFeature(),
        domain_classifier=classifier, device="cpu"
    )
    predictor.calibrate(
        DataLoader(TensorDataset(X, y), batch_size=batch_size), alpha=0.2
    )
    test = torch.tensor([[.05, .5, .95, 0.], [.05, .5, .95, 9.]])
    test[:, -1] = torch.tensor([1., 9.]).log()
    test_labels = torch.zeros(2, dtype=torch.long)
    metrics = predictor.evaluate(DataLoader(
        TensorDataset(test, test_labels), batch_size=batch_size
    ))

    # Compute each quantile directly from the score-weight pairs plus the
    # test point's probability mass at infinity.
    expected = []
    for test_weight in [1., 9.]:
        total = sum(cal_weights) + test_weight
        accumulated = 0.
        threshold = float("inf")
        for score, weight in sorted(zip(cal_scores, cal_weights)):
            accumulated += weight
            if accumulated / total >= 0.8:
                threshold = score
                break
        expected.append([int(s <= threshold) for s in [.05, .5, .95]])
    expected = torch.tensor(expected, dtype=torch.int)
    torch.testing.assert_close(predictor.predict(test), expected)
    assert metrics["average_size"] == expected.sum(1).float().mean().item()
