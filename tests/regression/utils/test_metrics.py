import pytest
import torch

from torchcp.regression.utils.metrics import coverage_rate, average_size, Metrics
from torchcp.utils.registry import Registry

# Register the test metrics
METRICS_REGISTRY_REGRESSION = Registry("METRICS")


# Test Data Setup
@pytest.fixture
def mock_data():
    # Test data: prediction intervals and ground truth labels
    prediction_intervals = torch.tensor([
        [0.1, 0.5, 0.2, 0.6],  # Prediction intervals (lower, upper)
        [0.3, 0.7, 0.4, 0.8],
    ])  # Shape: [batch_size, num_intervals * 2]
    y_truth = torch.tensor([0.45, 0.75])  # Ground truth labels
    return prediction_intervals, y_truth


def test_coverage_rate(mock_data):
    prediction_intervals, y_truth = mock_data

    # Call the coverage_rate function
    result = coverage_rate(prediction_intervals, y_truth)

    # Manually calculate the coverage rate
    condition = torch.zeros_like(y_truth, dtype=torch.bool)
    for i in range(prediction_intervals.shape[1] // 2):
        lower_bound = prediction_intervals[:, 2 * i]
        upper_bound = prediction_intervals[:, 2 * i + 1]
        condition |= (y_truth >= lower_bound) & (y_truth <= upper_bound)

    expected_coverage_rate = torch.sum(condition).cpu() / y_truth.shape[0]

    # Ensure the computed coverage rate is correct
    assert abs(
        result - expected_coverage_rate) <= 1e-1, f"Expected coverage_rate {expected_coverage_rate}, but got {result}"


def test_column_validation():
    # Test case 1: Valid even number of columns
    valid_intervals = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # 4 columns (2 intervals)
    assert coverage_rate(valid_intervals, torch.tensor([1.5])) >= 0

    # Test case 2: Invalid odd number of columns
    invalid_intervals = torch.tensor([[1.0, 2.0, 3.0]])  # 3 columns
    with pytest.raises(ValueError) as exc_info:
        coverage_rate(invalid_intervals, torch.tensor([1.5]))
    assert "must be even" in str(exc_info.value)

    # Test case 3: Another valid case with multiple intervals
    valid_multi = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])  # 6 columns (3 intervals)
    assert average_size(valid_multi) > 0

    # Test case 4: Invalid case for average_size
    invalid_avg = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])  # 5 columns
    with pytest.raises(ValueError) as exc_info:
        average_size(invalid_avg)
    assert "must be even" in str(exc_info.value)


def test_average_size():
    # Test data
    prediction_intervals = torch.tensor([
        [0.1, 0.5, 0.2, 0.6],  # Two intervals per sample
        [0.3, 0.7, 0.4, 0.8],
    ])

    # Call function
    result = average_size(prediction_intervals)

    expected = 0.8

    assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"


def test_average_size_invalid_shape():
    invalid_intervals = torch.tensor([0.1, 0.5, 0.2])  # Odd number of columns
    with pytest.raises(ValueError):
        average_size(invalid_intervals)


def test_metrics_class():
    metrics = Metrics()

    # Test if registered metrics return the correct function
    metric_function = metrics("coverage_rate")
    assert metric_function == coverage_rate, f"Expected 'coverage_rate', but got {metric_function}"

    metric_function = metrics("average_size")
    assert metric_function == average_size, f"Expected 'average_size', but got {metric_function}"

    # Test for an unregistered metric (should raise NameError)
    with pytest.raises(NameError):
        metrics("non_existing_metric")
