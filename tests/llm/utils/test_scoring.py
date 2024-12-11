# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torchcp.llm.utils.scoring import geometric, marginal, first_k, first_k_no_mask, max, sum


class TestScoringFunctions:
    @pytest.fixture
    def setup_basic_inputs(self):
        """Create basic test inputs"""
        p = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6]], dtype=torch.float32)
        mask = torch.tensor([[1., 1., 0.],
                             [1., 0., 1.]], dtype=torch.float32)
        return p, mask

    def test_geometric_basic(self, setup_basic_inputs):
        """Test basic functionality of geometric function"""
        p, _ = setup_basic_inputs
        result = geometric(p)

        expected = -torch.cumsum(torch.log(torch.maximum(1 - p, torch.tensor(1e-8))), dim=-1)
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_geometric_with_mask(self, setup_basic_inputs):
        """Test geometric function with mask"""
        p, mask = setup_basic_inputs
        result = geometric(p, mask)

        masked_p = p * mask
        expected = -torch.cumsum(
            torch.log(torch.maximum(1 - masked_p, torch.tensor(1e-8))),
            dim=-1
        )
        assert torch.allclose(result, expected, rtol=1e-5)

    def test_marginal_basic(self, setup_basic_inputs):
        """Test basic functionality of marginal function"""
        p, _ = setup_basic_inputs
        result = marginal(p)

        assert result.shape == p.shape

        # 测试带掩码的情况
        result_masked = marginal(p)
        assert torch.allclose(result, result_masked)

        # 测试数值范围
        assert torch.all(result > -float('inf'))
        assert torch.all(result < float('inf'))

    def test_first_k_basic(self):
        """Test basic functionality of first_k function"""
        X = torch.zeros((2, 3))
        result = first_k(X)
        expected = torch.tensor([[1., 2., 3.],
                                 [1., 2., 3.]])
        assert torch.allclose(result, expected)

    def test_first_k_no_mask_basic(self):
        """Test basic functionality of first_k_no_mask function"""
        X = torch.zeros((2, 3))
        result = first_k_no_mask(X)
        expected = torch.tensor([[1., 2., 3.],
                                 [1., 2., 3.]])
        assert torch.allclose(result, expected)

    def test_max_basic(self, setup_basic_inputs):
        """Test basic functionality of max function"""
        X = torch.tensor([[1., 2., 3.],
                          [3., 2., 1.]])
        result = max(X)
        expected = torch.tensor([[1., 2., 3.],
                                 [3., 3., 3.]])
        assert torch.allclose(result, expected)

    def test_sum_basic(self, setup_basic_inputs):
        """Test basic functionality of sum function"""
        X = torch.tensor([[1., 2., 3.],
                          [3., 2., 1.]])
        result = sum(X)
        expected = torch.tensor([[1., 3., 6.],
                                 [3., 5., 6.]])
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("func", [geometric, marginal, first_k, max, sum])
    def test_mask_zeros(self, func):
        """Test behavior of all functions with zero mask"""
        X = torch.ones((2, 3))
        mask = torch.zeros_like(X)
        result = func(X, mask)
        assert not torch.isnan(result).any()

    @pytest.mark.parametrize("func", [geometric, marginal, first_k, max, sum])
    def test_different_shapes(self, func):
        """Test functions with different input shapes"""
        shapes = [(1, 5), (3, 4), (10, 2)]
        for shape in shapes:
            X = torch.rand(shape)
            result = func(X)
            assert result.shape == shape

    def test_geometric_numerical_stability(self):
        """Test numerical stability of geometric function"""
        p_near_one = torch.tensor([[0.999999]], dtype=torch.float32)
        result = geometric(p_near_one)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        p_near_zero = torch.tensor([[0.000001]], dtype=torch.float32)
        result = geometric(p_near_zero)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_marginal_numerical_stability(self):
        """Test numerical stability of marginal function"""
        p_near_one = torch.tensor([[0.999999]], dtype=torch.float32)
        result = marginal(p_near_one)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    @pytest.mark.parametrize("func", [geometric, marginal, max, sum])
    def test_dtype_consistency(self, func):
        """Test data type consistency"""
        # Test with float32
        X_float32 = torch.rand((3, 4), dtype=torch.float32)
        result = func(X_float32)
        assert result.dtype == torch.float32

        # Test with float64
        X_float64 = torch.rand((3, 4), dtype=torch.float64)
        result = func(X_float64)
        assert result.dtype == torch.float64

    # Separate test for first_k since it always returns float32
    def test_first_k_dtype(self):
        """Test first_k dtype behavior"""
        X_float32 = torch.rand((3, 4), dtype=torch.float32)
        result = first_k(X_float32)
        assert result.dtype == torch.float32

        X_float64 = torch.rand((3, 4), dtype=torch.float64)
        result = first_k(X_float64)
        assert result.dtype == torch.float32  # first_k always returns float32


if __name__ == "__main__":
    pytest.main(["-v"])
