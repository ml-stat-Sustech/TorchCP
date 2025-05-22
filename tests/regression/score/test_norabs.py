import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torchcp.regression.score.norabs import NorABS


@pytest.fixture
def norabs_instance():
    return NorABS()


def test_call_with_2d_output(norabs_instance):
    predicts = torch.tensor([[0.5, 0.1], [0.4, 0.2]])
    y_truth = torch.tensor([0.6, 0.1])

    scores = norabs_instance(predicts, y_truth)
    expected_scores = torch.tensor([[1.0], [1.5]])
    assert torch.allclose(scores, expected_scores), "The __call__ output is incorrect."


def test_call_with_1d_scores(norabs_instance):
    # Ensure the unsqueeze works if the score is 1D
    predicts = torch.tensor([[0.5, 1.0]])
    y_truth = torch.tensor([0.0])
    scores = norabs_instance(predicts, y_truth)
    assert scores.shape == (1, 1), "The score should be 2D with shape (1, 1)."


def test_generate_intervals_single_threshold(norabs_instance):
    predicts = torch.tensor([[0.5, 0.1], [0.4, 0.2]])
    q_hat = torch.tensor([2.0])

    intervals = norabs_instance.generate_intervals(predicts, q_hat)
    expected = torch.tensor([[[0.3, 0.7]], [[0.0, 0.8]]])
    assert torch.allclose(intervals, expected), "Interval calculation failed for single threshold."


def test_generate_intervals_multi_threshold(norabs_instance):
    predicts = torch.tensor([[0.5, 0.1]])
    q_hat = torch.tensor([1.0, 2.0])

    intervals = norabs_instance.generate_intervals(predicts, q_hat)
    expected = torch.tensor([[[0.4, 0.6], [0.3, 0.7]]])
    assert torch.allclose(intervals, expected), "Interval calculation failed for multiple thresholds."


def test_train_returns_model(norabs_instance, dummy_data):
    train_dataloader, _ = dummy_data
    model = norabs_instance.train(train_dataloader, epochs=2, verbose=True)
    model = norabs_instance.train(train_dataloader, epochs=2, verbose=False)
    sample_input = next(iter(train_dataloader))[0]

    with torch.no_grad():
        output = model(sample_input)
    assert output.shape[1] == 2, "The trained model should output both mean and variance."
