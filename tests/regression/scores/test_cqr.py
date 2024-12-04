# test_cqr.py

import pytest
import torch

from torchcp.regression.score import CQR


@pytest.fixture
def cqr_instance():
    """
    Fixture to provide an instance of the CQR class.
    """
    return CQR()


def test_call(cqr_instance):
    """
    Test the __call__ method for score calculation.
    """
    predicts = torch.tensor([[0.2, 0.7], [0.3, 0.8]])
    y_truth = torch.tensor([0.5, 0.4])

    scores = cqr_instance(predicts, y_truth)
    expected_scores = torch.tensor([[-0.2], [-0.1]])
    assert torch.allclose(scores, expected_scores), "The __call__ method is not working as expected."


def test_generate_intervals(cqr_instance):
    """
    Test the generate_intervals method for prediction interval generation.
    """ 
    predicts_batch = torch.tensor([[0.2, 0.7], [0.3, 0.8]])
    q_hat = torch.tensor([0.1])

    intervals = cqr_instance.generate_intervals(predicts_batch, q_hat)
    expected_intervals = torch.tensor([[[0.1, 0.8]], [[0.2, 0.9]]])
    assert torch.allclose(intervals, expected_intervals), "The generate_intervals method is not working as expected."


def test_fit(cqr_instance, dummy_data):
    """
    Test the fit method to ensure the model trains correctly.
    """
    train_dataloader, _ = dummy_data
    with pytest.raises(ValueError):
        model = cqr_instance.fit(train_dataloader)
    model = cqr_instance.fit(train_dataloader, alpha=0.1, epochs=5, verbose=False)

    # Check model output shape
    test_input = next(iter(train_dataloader))[0]
    with torch.no_grad():
        output = model(test_input)
    assert output.shape[1] == 2, "The model output shape is incorrect."
