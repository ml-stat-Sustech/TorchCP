# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from torchcp.classification.trainer.ordinal_trainer import OrdinalClassifier, \
    OrdinalTrainer  # Replace with the actual import path



# Mock model for testing
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # 5 ordinal classes
        
    def forward(self, x):
        return self.linear(x)

# Fixtures
@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_data():
    # Create mock dataset
    x = torch.randn(500, 10)  # 500 samples, 10 features
    y = torch.randint(0, 5, (500,))  # 5 ordinal classes
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

@pytest.fixture
def ordinal_config():
    return {
        "phi": "abs",
        "varphi": "abs"
    }

@pytest.fixture
def ordinal_trainer(mock_model, ordinal_config):
    return OrdinalTrainer(
        ordinal_config=ordinal_config,
        model=mock_model
    )

# Tests
class TestOrdinalTrainer:
    def test_initialization(self, mock_model, ordinal_config):
        trainer = OrdinalTrainer(
            ordinal_config=ordinal_config,
            model=mock_model
        )
        
    def test_invalid_config(self, mock_model):
        # Test invalid phi
        invalid_config = {
            "phi": "invalid",
            "varphi": "abs"
        }
        with pytest.raises(NotImplementedError):
            OrdinalTrainer(
                ordinal_config=invalid_config,
                model=mock_model
            )
            
        # Test invalid varphi
        invalid_config = {
            "phi": "abs",
            "varphi": "invalid"
        }
        with pytest.raises(NotImplementedError):
            OrdinalTrainer(
                ordinal_config=invalid_config,
                model=mock_model
            )
            
    def test_different_configs(self, mock_model, mock_data):
        # Test all valid combinations of phi and varphi
        configs = [
            {"phi": "abs", "varphi": "abs"},
            {"phi": "abs", "varphi": "square"},
            {"phi": "square", "varphi": "abs"},
            {"phi": "square", "varphi": "square"}
        ]
        
        for config in configs:
            trainer = OrdinalTrainer(
                ordinal_config=config,
                model=mock_model
            )
            trainer.train(mock_data, num_epochs=1)
            
    def test_training_process(self, ordinal_trainer, mock_data):
        # Test basic training
        ordinal_trainer.train(
            train_loader=mock_data,
            val_loader=mock_data,  # Using same data for simplicity
            num_epochs=2
        )
        
    def test_gpu_training(self, mock_model, ordinal_config, mock_data):
        if torch.cuda.is_available():
            trainer = OrdinalTrainer(
                ordinal_config=ordinal_config,
                model=mock_model,
                device=torch.device('cuda')
            )
            trainer.train(mock_data, num_epochs=2)