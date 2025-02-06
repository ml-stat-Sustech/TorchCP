import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torchcp.classification.trainer.ts_trainer import TSTrainer, _ECELoss



# Mock model
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
    x = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16)

@pytest.fixture
def ece_loss():
    return _ECELoss(n_bins=5)

@pytest.fixture
def ts_trainer(mock_model):
    return TSTrainer(init_temperature=1.0, model=mock_model)

# Tests for _ECELoss
class TestECELoss:
    def test_initialization(self):
        criterion = _ECELoss(n_bins=10)
        
    def test_forward_pass(self, ece_loss):
        logits = torch.randn(10, 3)
        labels = torch.randint(0, 3, (10,))
        _ = ece_loss(logits, labels)
        
    def test_device_handling(self):
        if torch.cuda.is_available():
            criterion = _ECELoss().cuda()
            logits = torch.randn(10, 2).cuda()
            labels = torch.randint(0, 2, (10,)).cuda()
            _ = criterion(logits, labels)

# Tests for TSTrainer
class TestTSTrainer:
    def test_initialization(self, mock_model):
        trainer = TSTrainer(init_temperature=1.5, model=mock_model)
        
    def test_invalid_temperature(self, mock_model):
        with pytest.raises(ValueError):
            TSTrainer(init_temperature=-1.0, model=mock_model)
        
    def test_training_process(self, ts_trainer, mock_data):
        # Train model
        ts_trainer.train(mock_data, lr=0.01, num_epochs=2)
        
    def test_gpu_training(self, mock_model, mock_data):
        if torch.cuda.is_available():
            trainer = TSTrainer(
                init_temperature=1.0, 
                model=mock_model,
                device=torch.device('cuda')
            )
            trainer.train(mock_data, num_epochs=2)