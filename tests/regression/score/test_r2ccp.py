# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.regression.score import R2CCP
from torchcp.regression.utils import calculate_midpoints


@pytest.fixture
def r2ccp_instance(dummy_data):
    """
    Fixture to provide an instance of the R2CCP class.
    """
    K = 3
    train_dataloader, _ = dummy_data
    midpoints = calculate_midpoints(train_dataloader, K)
    return R2CCP(midpoints)


def test_call(r2ccp_instance):
    """
    Test the __call__ method for score calculation.
    """
    predicts = torch.tensor([[0.2, 0.7, 0.9], [0.3, 0.8, 1.0]])
    y_truth = torch.tensor([0.5, 0.4])

    scores = r2ccp_instance(predicts, y_truth)
    expected_scores = torch.tensor([[-0.7], [-0.7]])
    assert torch.all(
        torch.isclose(scores, expected_scores, atol=1e-1)), "The __call__ method is not working as expected."


def test_generate_intervals(r2ccp_instance):
    """
    Test the generate_intervals method for prediction interval generation.
    """
    predicts_batch = torch.tensor([[0.2, 0.7, 0.9], [0.3, 0.8, 1.0]])
    q_hat = torch.tensor([0.1])

    intervals = r2ccp_instance.generate_intervals(predicts_batch, q_hat)
    expected_intervals = torch.tensor([[0, 0.5, 0.5, 1.0], [0, 0.5, 0.5, 1.0]])
    assert torch.all(torch.isclose(intervals, expected_intervals,
                                   atol=1e-1)), "The generate_intervals method is not working as expected."


def test_train(r2ccp_instance, dummy_data):
    """
    Test the train method to ensure the model trains correctly.
    """
    train_dataloader, _ = dummy_data
    model = r2ccp_instance.train(train_dataloader, alpha=0.1, epochs=5, verbose=False)

    # Check model output shape
    test_input = next(iter(train_dataloader))[0]
    with torch.no_grad():
        output = model(test_input)
    assert output.shape[1] == 3, "The model output shape is incorrect."
