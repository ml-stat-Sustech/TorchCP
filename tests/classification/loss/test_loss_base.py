import pytest
import torch
import torch.nn as nn
from torchcp.classification.loss.base import BaseLoss

import pytest
import torch
import torch.nn as nn

class MockPredictor:
    pass

class MockLoss(BaseLoss):
    def forward(self, predictions, targets):
        return predictions - targets

@pytest.fixture
def mock_loss_instance():
    weight = 1.0
    predictor = MockPredictor()
    return MockLoss(weight, predictor)

def test_init(mock_loss_instance):
    mock_loss = mock_loss_instance
    assert mock_loss.weight == 1.0
    assert isinstance(mock_loss.predictor, MockPredictor)

def test_invalid_weight():
    with pytest.raises(ValueError):
        MockLoss(0, MockPredictor())

def test_forward_not_implemented():
    base_loss = BaseLoss(1.0, MockPredictor())
    with pytest.raises(NotImplementedError):
        base_loss.forward(None, None)

def test_forward(mock_loss_instance):
    mock_loss = mock_loss_instance
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    output = mock_loss.forward(predictions, targets)
    assert torch.equal(output, torch.tensor([0.0, 0.0, 0.0]))

if __name__ == '__main__':
    pytest.main()
if __name__ == "__main__":
    pytest.main(["-v", "--cov=base", "--cov-report=term-missing"])