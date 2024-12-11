# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest
import torch
import torch.nn as nn

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


from unittest.mock import patch


def test_get_device_none_with_cuda():
    """Test when model is None and cuda is available"""
    with patch('torch.cuda.is_available', return_value=True), \
            patch('torch.cuda.current_device', return_value=0):
        device = get_device(None)
        assert device == torch.device('cuda:0')


def test_get_device_none_without_cuda():
    """Test when model is None and cuda is not available"""
    with patch('torch.cuda.is_available', return_value=False):
        device = get_device(None)
        assert device == torch.device('cpu')


def test_get_device_with_cpu_model():
    """Test with a model on CPU"""
    model = nn.Linear(10, 5)  # Creates a model on CPU by default
    device = get_device(model)
    assert device == torch.device('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Skip if CUDA is not available")
def test_get_device_with_gpu_model():
    """Test with a model on GPU (only runs if CUDA is available)"""
    model = nn.Linear(10, 5).cuda()
    device = get_device(model)
    assert device == torch.device(f'cuda:{torch.cuda.current_device()}')


def test_get_device_with_specific_gpu():
    """Test with a model on a specific GPU device"""
    # Create a mock model with a parameter that appears to be on cuda:1
    model = nn.Linear(10, 5)
    mock_device = torch.device('cuda:1')
    # Mock the next() and parameters() calls to return a tensor with our desired device
    with patch.object(model, 'parameters') as mock_parameters:
        mock_param = torch.nn.Parameter(torch.randn(1))
        mock_param.data = mock_param.data.to(mock_device)
        mock_parameters.return_value = iter([mock_param])
        device = get_device(model)
        assert device == mock_device


def test_default_q_hat_max_normal_case():
    """Test when default_q_hat is 'max' and scores is not empty"""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.1
    result = calculate_conformal_value(scores, alpha, default_q_hat="max")

    assert result == 5.0


def test_calculate_conformal_value():
    """Test conformal value calculation"""
    # Test normal case
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    alpha = 0.2
    result = calculate_conformal_value(scores, alpha)
    assert torch.allclose(result, torch.tensor(5.0))

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
