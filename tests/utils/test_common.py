import pytest
import torch
import numpy as np
from torchcp.utils.common import get_device, calculate_conformal_value, DimensionError


def test_dimension_error():
    with pytest.raises(DimensionError):
        raise DimensionError("This is a dimension error")

def test_dimension_error_message():
    try:
        raise DimensionError("This is a dimension error")
    except DimensionError as e:
        assert str(e) == "This is a dimension error"
        
        
class DummyModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1).to(device)

def test_get_device():
    """Test device detection logic"""
    # Test with no model
    device = get_device(None)
    expected_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert device == expected_device
    
    # Test with CPU model
    model_cpu = DummyModel("cpu")
    assert get_device(model_cpu) == torch.device("cpu")
    
    # Test with CUDA model if available
    if torch.cuda.is_available():
        model_cuda = DummyModel("cuda")
        assert get_device(model_cuda) == torch.device("cuda:0")

def test_calculate_conformal_value():
    """Test conformal value calculation"""
    # Test normal case
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.2
    result = calculate_conformal_value(scores, alpha)
    assert torch.allclose(result, torch.tensor(4.0))
    
    # Test invalid alpha values
    with pytest.raises(ValueError, match="Significance level 'alpha' must be in"):
        calculate_conformal_value(scores, alpha=1.5)
    
    with pytest.raises(ValueError, match="Significance level 'alpha' must be in"):
        calculate_conformal_value(scores, alpha=0)
    
    # Test empty scores
    with pytest.warns(UserWarning, match="The number of scores is 0"):
        result = calculate_conformal_value(torch.tensor([]), alpha=0.1)
        assert result == torch.inf
    
    # Test quantile value exceeding 1
    small_scores = torch.tensor([1.0])
    with pytest.warns(UserWarning, match="The value of quantile exceeds 1"):
        result = calculate_conformal_value(small_scores, alpha=0.1)
        assert result == torch.inf
    
    # Test custom default value
    with pytest.warns(UserWarning):
        result = calculate_conformal_value(torch.tensor([]), alpha=0.1, default_q_hat=torch.tensor(999))
        assert result == 999
    
    # Test device consistency
    if torch.cuda.is_available():
        cuda_scores = scores.cuda()
        result = calculate_conformal_value(cuda_scores, alpha=0.2)
        assert result.device == cuda_scores.device
