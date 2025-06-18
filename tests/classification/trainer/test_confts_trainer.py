# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from torchcp.classification.loss import ConfTSLoss
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS
from torchcp.classification.trainer import ConfTSTrainer, TSTrainer
from torchcp.classification.trainer import TemperatureScalingModel



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
    return ConfTSTrainer(
        model=mock_model,
        init_temperature=1.0,
        alpha=0.1
    )

# Tests
class TestConfTSTrainer:
    def test_initialization(self, mock_model):
        trainer = ConfTSTrainer(
            model=mock_model,
            init_temperature=1.5,
            alpha=0.1
        )
        
    def test_invalid_params(self, mock_model):
        # Test invalid temperature
        with pytest.raises(ValueError):
            ConfTSTrainer(
                model=mock_model,
                init_temperature=-1.0,
                alpha=0.1
            )
            
        # Test invalid alpha
        with pytest.raises(ValueError):
            ConfTSTrainer(
                model=mock_model,
                init_temperature=1.0,
                alpha=-0.1
            )
            
    def test_training_process(self, confts_trainer, mock_data):
        # Train model with default parameters
        confts_trainer.train(mock_data, num_epochs=2)
        
    def test_gpu_training(self, mock_model, mock_data):
        if torch.cuda.is_available():
            trainer = ConfTSTrainer(
                model=mock_model,
                init_temperature=1.0,
                alpha=0.1,
                device=torch.device('cuda')
            )
            trainer.train(mock_data, num_epochs=2)
            
    def test_training_with_different_params(self, confts_trainer, mock_data):
        # Test training with different learning rates and epochs
        confts_trainer.train(mock_data, lr=0.001, num_epochs=1)
        confts_trainer.train(mock_data, lr=0.1, num_epochs=1)