# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.regression.loss import R2ccpLoss


@pytest.fixture
def setup_loss():
    """Fixture to set up the loss function and default parameters."""
    midpoints = torch.tensor([0.1, 0.5, 0.9])
    return R2ccpLoss(p=2, tau=0.5, midpoints=midpoints)


def test_loss_computation(setup_loss):
    """Test basic loss computation."""
    loss_fn = setup_loss
    preds = torch.tensor([[0.2, 0.5, 0.3],
                          [0.1, 0.7, 0.2],
                          [0.3, 0.4, 0.3]], requires_grad=True).softmax(dim=1)
    target = torch.tensor([[0.2], [0.6], [0.8]])

    loss = loss_fn(preds, target)
    expected_loss = torch.tensor(2.1)
    assert loss.requires_grad, "Loss should require gradients."
    assert loss.item() > 0, "Loss should be positive."
    assert torch.isclose(loss, expected_loss, atol=1e-1), f"Loss mismatch: {loss} != {expected_loss}"


def test_batch_size_mismatch(setup_loss):
    """Test handling of batch size mismatch."""
    loss_fn = setup_loss
    preds = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]]).softmax(dim=1)
    target = torch.tensor([[0.2]])

    with pytest.raises(IndexError, match="Batch size mismatch"):
        loss_fn(preds, target)
