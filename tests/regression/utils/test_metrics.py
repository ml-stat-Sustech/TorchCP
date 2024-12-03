import pytest
import torch
from torchcp.utils.registry import Registry
from torchcp.regression.utils.metrics import coverage_rate, average_size, Metrics

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
    assert torch.allclose(result, expected_coverage_rate, atol=1e-1), f"Expected coverage_rate {expected_coverage_rate}, but got {result}"


def test_average_size(mock_data):
    prediction_intervals, _ = mock_data

    # Call the average_size function
    result = average_size(prediction_intervals)

    # Ensure the result is a scalar tensor
    assert result.shape == torch.Size([]), f"Expected result shape [], but got {result.shape}"

    # Manually compute the average size of prediction intervals
    size = torch.abs(prediction_intervals[..., 1::2] - prediction_intervals[..., 0::2]).sum(dim=-1)
    expected_average_size = size.mean(dim=0).cpu()

    # Ensure the computed average size is correct
    assert torch.allclose(result, expected_average_size, atol=1e-6), f"Expected average_size {expected_average_size}, but got {result}"


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
