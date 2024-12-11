# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.regression.score import CQRR


@pytest.fixture
def cqrr_instance():
    """
    Fixture to provide an instance of the CQRR class.
    """
    return CQRR()


def test_call(cqrr_instance):
    """
    Test the __call__ method for score calculation.
    """
    predicts = torch.tensor([[0.2, 0.7], [0.3, 0.8]])
    y_truth = torch.tensor([0.5, 0.4])

    scores = cqrr_instance(predicts, y_truth)
    expected_scores = torch.tensor([[-0.4], [-0.2]])
    assert torch.allclose(scores, expected_scores), "The __call__ method is not working as expected."


def test_generate_intervals(cqrr_instance):
    """
    Test the generate_intervals method for prediction interval generation.
    """
    predicts_batch = torch.tensor([[0.2, 0.7], [0.3, 0.8]])
    q_hat = torch.tensor([0.1])

    intervals = cqrr_instance.generate_intervals(predicts_batch, q_hat)
    expected_intervals = torch.tensor([[[0.2, 0.7]], [[0.3, 0.8]]])
    assert torch.allclose(intervals, expected_intervals), "The generate_intervals method is not working as expected."


def test_fit(cqrr_instance, dummy_data):
    """
    Test the train method to ensure the model trains correctly.
    """
    train_dataloader, _ = dummy_data
    model = cqrr_instance.train(train_dataloader, alpha=0.1, epochs=5, verbose=False)

    # Check model output shape
    test_input = next(iter(train_dataloader))[0]
    with torch.no_grad():
        output = model(test_input)
    assert output.shape[1] == 2, "The model output shape is incorrect."
