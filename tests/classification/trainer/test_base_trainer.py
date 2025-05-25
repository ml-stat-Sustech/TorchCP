# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tempfile
import os

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import tempfile

from torchcp.classification.trainer.base_trainer import BaseTrainer, Trainer

# Mock model for testing
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

# Concrete implementation of BaseTrainer for testing abstract class
class ConcreteTrainer(BaseTrainer):
    def train(self, train_loader, val_loader=None, **kwargs):
        return self.model

# Fixtures
@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture
def mock_data():
    # Create mock dataset
    x = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.randint(0, 2, (100,))  # Binary classification
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16)

@pytest.fixture
def base_trainer(mock_model):
    return ConcreteTrainer(mock_model)

@pytest.fixture
def trainer(mock_model):
    return Trainer(mock_model)

# Tests for BaseTrainer
class TestBaseTrainer:
    def test_train_abstract_method(self):
        class FailTrainer(BaseTrainer):
            pass  
        
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FailTrainer(MockModel())
            
    def test_train_implementation(self, base_trainer, mock_data):
        model = base_trainer.train(mock_data)
        assert isinstance(model, torch.nn.Module)
        
        model = base_trainer.train(mock_data, mock_data)
        assert isinstance(model, torch.nn.Module)
        
        model = base_trainer.train(mock_data, extra_param=True)
        assert isinstance(model, torch.nn.Module)
        
        
    def test_initialization(self, mock_model):
        # Test with default device
        trainer = ConcreteTrainer(mock_model)
        assert trainer.device == torch.device('cpu')
        
        # Test with specified device
        cpu_trainer = ConcreteTrainer(mock_model, device=torch.device('cpu'))
        assert cpu_trainer.device == torch.device('cpu')
        
        # Test verbose setting
        assert trainer.verbose == True
        quiet_trainer = ConcreteTrainer(mock_model, verbose=False)
        assert quiet_trainer.verbose == False

    def test_invalid_initialization(self):
        with pytest.raises(ValueError, match="Model cannot be None"):
            ConcreteTrainer(None)

    def test_model_save_load(self, base_trainer):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            try:
                # Save model
                base_trainer.save_model(tmp.name)
                assert os.path.exists(tmp.name)
                
                # Change model parameters
                original_params = {name: param.clone() for name, param in base_trainer.model.named_parameters()}
                for param in base_trainer.model.parameters():
                    nn.init.constant_(param, 0.0)
                
                # Load model and verify parameters are restored
                base_trainer.load_model(tmp.name)
                for name, param in base_trainer.model.named_parameters():
                    assert torch.allclose(param, original_params[name])
            finally:
                os.unlink(tmp.name)


    def test_train_pass_statement(self):
        """直接测试基类中train方法的pass语句"""
        class TempTrainer(BaseTrainer):
            def train(self, train_loader, val_loader=None, **kwargs):
                return super().train(train_loader, val_loader, **kwargs)
        
        model = MockModel()
        trainer = TempTrainer(model)
        data_loader = DataLoader(TensorDataset(torch.randn(10, 10), torch.zeros(10)))
        
        result = BaseTrainer.train.__get__(trainer)(data_loader)
        
        assert result is None
    
    
# Tests for Trainer
class TestTrainer:
    def test_initialization(self, trainer):
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)

    def test_calculate_loss(self, trainer):
        output = torch.randn(4, 2)  # 4 samples, 2 classes
        target = torch.tensor([0, 1, 0, 1])
        loss = trainer.calculate_loss(output, target)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar tensor

    def test_train_epoch(self, trainer, mock_data):
        loss = trainer.train_epoch(mock_data)
        assert isinstance(loss, float)
        assert loss > 0  # Loss should be positive

    def test_validate(self, trainer, mock_data):
        loss = trainer.validate(mock_data)
        assert isinstance(loss, float)
        assert loss > 0

    def test_full_training(self, trainer, mock_data):
        # Test training with validation
        trained_model = trainer.train(
            train_loader=mock_data,
            val_loader=mock_data,
            num_epochs=2
        )
        assert isinstance(trained_model, nn.Module)
        
        # Test training without validation
        trained_model = trainer.train(
            train_loader=mock_data,
            num_epochs=2
        )
        assert isinstance(trained_model, nn.Module)

    def test_model_save_load(self, trainer):
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            try:
                # Get initial parameters
                initial_params = {name: param.clone() for name, param in trainer.model.named_parameters()}
                
                # Save model
                trainer.save_model(tmp.name)
                assert os.path.exists(tmp.name)
                
                # Change parameters
                for param in trainer.model.parameters():
                    nn.init.constant_(param, 0.0)
                
                # Load and verify
                trainer.load_model(tmp.name)
                for name, param in trainer.model.named_parameters():
                    assert torch.allclose(param, initial_params[name])
            finally:
                os.unlink(tmp.name)

    def test_training_with_early_stopping(self, trainer, mock_data):
        """Test that the best model is saved during training with validation"""
        initial_state = {name: param.clone() for name, param in trainer.model.named_parameters()}
        
        # Train with validation to trigger best model saving
        trainer.train(
            train_loader=mock_data,
            val_loader=mock_data,
            num_epochs=3
        )
        
        # Verify model parameters have changed
        current_state = {name: param.clone() for name, param in trainer.model.named_parameters()}
        any_param_changed = False
        for name in initial_state:
            if not torch.allclose(initial_state[name], current_state[name]):
                any_param_changed = True
                break
        assert any_param_changed, "Model parameters should have changed during training"
        


