# Copyright (c) 2023-present, SUSTech-ML.
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
