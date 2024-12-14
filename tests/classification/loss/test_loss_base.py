# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import pytest
import torch
import torch
import torch.nn as nn
import torch.nn as nn

from torchcp.classification.loss.base import BaseLoss


class MockPredictor:
    pass


class MockLoss(BaseLoss):
    def forward(self, predictions, targets):
        return predictions - targets


@pytest.fixture
def mock_loss_instance():
    weight = 1.0
    predictor = MockPredictor()
    return MockLoss(predictor)


def test_init(mock_loss_instance):
    mock_loss = mock_loss_instance
    assert isinstance(mock_loss.predictor, MockPredictor)


def test_forward_not_implemented():
    base_loss = BaseLoss(MockPredictor())
    with pytest.raises(NotImplementedError):
        base_loss.forward(None, None)


def test_forward(mock_loss_instance):
    mock_loss = mock_loss_instance
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    output = mock_loss.forward(predictions, targets)
    assert torch.equal(output, torch.tensor([0.0, 0.0, 0.0]))

