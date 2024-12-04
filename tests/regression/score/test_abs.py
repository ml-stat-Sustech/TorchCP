# test_split.py

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchcp.regression.score import ABS

@pytest.fixture
def split_instance():
    """
    Fixture to provide an instance of the split class.
    """
    return ABS()

def test_call(split_instance):
    """
    Test the __call__ method for score calculation.
    """
    predicts = torch.tensor([[0.2], [0.3]])
    y_truth = torch.tensor([[0.5], [0.4]])
    
    scores = split_instance(predicts, y_truth)
    expected_scores = torch.tensor([[0.3], [0.1]])
    assert torch.allclose(scores, expected_scores), "The __call__ method is not working as expected."

def test_generate_intervals(split_instance):
    """
    Test the generate_intervals method for prediction interval generation.
    """
    predicts_batch = torch.tensor([[0.2], [0.8]])
    q_hat = torch.tensor([0.1])
    
    intervals = split_instance.generate_intervals(predicts_batch, q_hat)
    expected_intervals = torch.tensor([[[0.1, 0.3]], [[0.7, 0.9]]])
    assert torch.allclose(intervals, expected_intervals), "The generate_intervals method is not working as expected."

def test_fit(split_instance, dummy_data):
    """
    Test the fit method to ensure the model trains correctly.
    """
    train_dataloader, _ = dummy_data
    model = split_instance.fit(train_dataloader, epochs=5, verbose=False)
    
    # Check model output shape
    test_input = next(iter(train_dataloader))[0]
    with torch.no_grad():
        output = model(test_input)
    assert output.shape[1] == 1, "The model output shape is incorrect."
