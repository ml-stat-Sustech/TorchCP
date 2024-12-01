# test_cqrfm.py

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcp.regression.scores import CQRFM

@pytest.fixture
def dummy_data():
    """
    Fixture to provide dummy data for testing.
    """
    x_train = torch.rand((100, 10))
    y_train = torch.rand((100, 1))
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    x_test = torch.rand((20, 10))
    y_test = torch.rand((20, 1))
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    return train_dataloader, test_dataloader


@pytest.fixture
def cqrfm_instance():
    """
    Fixture to provide an instance of the CQRFM class.
    """
    return CQRFM()


def test_call(cqrfm_instance):
    """
    Test the __call__ method for score calculation.
    """
    predicts = torch.tensor([[[0.2, 0.45, 0.7]], [[0.3, 0.55, 0.8]]])
    y_truth = torch.tensor([[0.5], [0.4]])

    scores = cqrfm_instance(predicts, y_truth)
    expected_scores = torch.tensor([[0.2], [0.6]])
    assert torch.allclose(scores, expected_scores), "The __call__ method is not working as expected."


def test_generate_intervals(cqrfm_instance):
    """
    Test the generate_intervals method for prediction interval generation.
    """
    predicts_batch = torch.tensor([[[0.2, 0.4, 0.7]], [[0.3, 0.6, 0.8]]])
    q_hat = torch.tensor([0.1])

    intervals = cqrfm_instance.generate_intervals(predicts_batch, q_hat)
    expected_intervals = torch.tensor([[[0.38, 0.43]], [[0.57, 0.62]]])
    assert torch.allclose(intervals, expected_intervals), "The generate_intervals method is not working as expected."


def test_fit(cqrfm_instance, dummy_data):
    """
    Test the fit method to ensure the model trains correctly.
    """
    train_dataloader, _ = dummy_data
    model = cqrfm_instance.fit(train_dataloader, alpha=0.1, epochs=5, verbose=False)

    # Check model output shape
    test_input = next(iter(train_dataloader))[0]
    with torch.no_grad():
        output = model(test_input)
    assert output.shape[1] == 3, "The model output shape is incorrect."
