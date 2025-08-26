# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.regression.score import Sign


@pytest.fixture
def split_instance():
    """
    Fixture to provide an instance of the split class.
    """
    return Sign()


def test_call(split_instance):
    """
    Test the __call__ method for score calculation.
    """
    predicts = torch.tensor([[0.2], [0.3]])
    y_truth = torch.tensor([[0.5], [0.4]])

    scores = split_instance(predicts, y_truth)
    expected_scores = torch.tensor([[0.3], [0.1]])
    assert torch.allclose(scores, expected_scores), "The __call__ method is not working as expected."
