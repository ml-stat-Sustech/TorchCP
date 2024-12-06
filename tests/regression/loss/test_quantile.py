import pytest
import torch

from torchcp.regression.loss import QuantileLoss


def test_quantile_loss_computation():
    """Test the forward computation of QuantileLoss."""
    # Initialize QuantileLoss
    quantiles = [0.1, 0.5, 0.9]
    loss_fn = QuantileLoss(quantiles=quantiles)

    # Mock predictions and target
    preds = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]], requires_grad=True)
    target = torch.tensor([[0.3], [0.4]])

    # Compute the loss
    loss = loss_fn(preds, target)
    expected_loss = 0.17

    assert torch.isclose(loss, torch.tensor(expected_loss), atol=1e-4), f"Loss mismatch: {loss} != {expected_loss}"


def test_batch_size_mismatch():
    """Test that a batch size mismatch raises an assertion error."""
    quantiles = [0.1, 0.5, 0.9]
    loss_fn = QuantileLoss(quantiles=quantiles)

    preds = torch.tensor([[0.1, 0.5, 0.9]])
    target = torch.tensor([[0.3], [0.4]])

    with pytest.raises(ValueError,
                       match=f"Batch size mismatch between predictions and targets. Got preds: {preds.size(0)}, target: {target.size(0)}"):
        loss_fn(preds, target)


def test_zero_quantile():
    """Test edge case where a quantile is 0."""
    quantiles = [0.0]
    loss_fn = QuantileLoss(quantiles=quantiles)

    preds = torch.tensor([[0.1]], requires_grad=True)
    target = torch.tensor([[0.3]])

    loss = loss_fn(preds, target)
    expected_loss = torch.tensor(0.0, requires_grad=True)
    assert torch.isclose(loss, expected_loss, atol=1e-4), f"Loss mismatch for zero quantile: {loss} != {expected_loss}"
