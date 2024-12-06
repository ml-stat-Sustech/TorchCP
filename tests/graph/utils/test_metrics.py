import pytest
import torch

from torchcp.graph.utils import Metrics


@pytest.fixture
def mock_prediction_sets():
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 0]
    ])


@pytest.fixture
def mock_labels():
    return torch.tensor([0, 2, 1, 3])


@pytest.fixture
def metrics():
    return Metrics()


def test_coverage_rate(mock_prediction_sets, mock_labels, metrics):
    result = metrics('coverage_rate')(mock_prediction_sets, mock_labels)
    expected_coverage = 3 / 4
    assert result == expected_coverage


def test_average_size(mock_prediction_sets, mock_labels, metrics):
    result = metrics('average_size')(mock_prediction_sets, mock_labels)
    expected_avg_size = 3 / 2
    assert result == expected_avg_size


def test_singleton_hit_ratio(mock_prediction_sets, mock_labels, metrics):
    result = metrics('singleton_hit_ratio')(mock_prediction_sets, mock_labels)
    expected_ratio = 1 / 4
    assert result == expected_ratio


def test_metrics_registry(metrics):
    with pytest.raises(NameError, match="is not defined in TorchCP"):
        metrics("undefined_metric")


def test_empty_prediction_sets(metrics):
    with pytest.raises(AssertionError, match="The number of prediction set must be greater than 0."):
        metrics('singleton_hit_ratio')([], [])
