# test_cqrm.py

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcp.regression.score import CQRM


@pytest.fixture
def cqrm_instance():
    """
    Fixture to provide an instance of the CQRM class.
    """
    return CQRM()


def test_call(cqrm_instance):
    """
    Test the __call__ method for score calculation.
    """
    predicts = torch.tensor([[0.2, 0.45, 0.7], [0.3, 0.55, 0.8]])
    y_truth = torch.tensor([0.5, 0.4])

    scores = cqrm_instance(predicts, y_truth)
    expected_scores = torch.tensor([[-0.8], [-0.4]])
    assert torch.allclose(scores, expected_scores), "The __call__ method is not working as expected."


def test_generate_intervals(cqrm_instance):
    """
    Test the generate_intervals method for prediction interval generation.
    """
    predicts_batch = torch.tensor([[0.2, 0.4, 0.7], [0.3, 0.6, 0.8]])
    q_hat = torch.tensor([0.1])

    intervals = cqrm_instance.generate_intervals(predicts_batch, q_hat)
    expected_intervals = torch.tensor([[[0.18, 0.73]], [[0.27, 0.82]]])
    assert torch.allclose(intervals, expected_intervals), "The generate_intervals method is not working as expected."


def test_train(cqrm_instance, dummy_data):
    """
    Test the train method to ensure the model trains correctly.
    """
    train_dataloader, _ = dummy_data
    with pytest.raises(ValueError):
        model = cqrm_instance.train(train_dataloader)
    model = cqrm_instance.train(train_dataloader, alpha=0.1, epochs=5, verbose=False)

    # Check model output shape
    test_input = next(iter(train_dataloader))[0]
    with torch.no_grad():
        output = model(test_input)
    assert output.shape[1] == 3, "The model output shape is incorrect."
