# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchcp.classification.trainer import ConfTrTrainer



# Mock model for testing
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)
        
    def forward(self, x):
        return self.linear(x)

# Fixtures
@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_data():
    x = torch.randn(1000, 10)
    y = torch.randint(0, 3, (1000,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=128, shuffle=True)

@pytest.fixture
def confts_trainer(mock_model):
    return ConfTrTrainer(
        alpha=0.1,
        model=mock_model
    )

# Tests
class TestConfTrTrainer:
    def test_initialization(self, mock_model):
        trainer = ConfTrTrainer(
            alpha=0.1,
            model=mock_model
        )
        
    def test_invalid_params(self, mock_model):
            
        # Test invalid alpha
        with pytest.raises(ValueError):
            ConfTrTrainer(
                alpha=-0.1,
                model=mock_model
            )