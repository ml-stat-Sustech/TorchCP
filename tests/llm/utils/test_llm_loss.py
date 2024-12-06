import pytest
import torch

from torchcp.llm.utils.loss import set_losses_from_labels


def test_set_losses_from_labels():
    # Test basic case
    labels = torch.tensor([[1, 0, 1],
                           [0, 1, 0]], dtype=torch.float32)
    result = set_losses_from_labels(labels)
    expected = torch.tensor([[0, 0, 0],
                             [1, 0, 0]], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_edge_cases():
    # Single value
    labels = torch.tensor([1], dtype=torch.float32)
    result = set_losses_from_labels(labels)
    assert torch.allclose(result, torch.tensor([0], dtype=torch.float32))

    # All zeros
    labels = torch.zeros(3, dtype=torch.float32)
    result = set_losses_from_labels(labels)
    assert torch.allclose(result, torch.ones(3, dtype=torch.float32))

    # All ones
    labels = torch.ones(3, dtype=torch.float32)
    result = set_losses_from_labels(labels)
    assert torch.allclose(result, torch.zeros(3, dtype=torch.float32))


def test_shapes():
    # 2D input
    labels = torch.randint(0, 2, (5, 3)).float()
    result = set_losses_from_labels(labels)
    assert result.shape == labels.shape

    # 3D input
    labels = torch.randint(0, 2, (2, 3, 4)).float()
    result = set_losses_from_labels(labels)
    assert result.shape == labels.shape


if __name__ == "__main__":
    pytest.main(["-v"])
