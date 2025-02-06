# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import pytest
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,TensorDataset

from torchcp.classification.loss import UncertaintyAwareLoss
from torchcp.classification.trainer.ua_trainer import UncertaintyAwareTrainer, TrainDataset

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
    # Create mock dataset
    x = torch.randn(500, 10)  # 500 samples, 10 features
    y = torch.randint(0, 3, (500,))  # 3 classes
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

@pytest.fixture
def uncertainty_trainer(mock_model):
    return UncertaintyAwareTrainer(
        weight=0.5,
        model=mock_model
    )

# Tests for TrainDataset
class TestTrainDataset:
    def test_train_dataset(self):
        X = torch.randn(100, 10)
        Y = torch.randint(0, 3, (100,))
        Z = torch.zeros(100)
        
        dataset = TrainDataset(X, Y, Z)
        assert len(dataset) == 100
        
        x, y, z = dataset[0]
        assert x.shape == torch.Size([10])
        assert isinstance(y.item(), int)
        assert z.item() in [0, 1]

# Tests for UncertaintyAwareTrainer
class TestUncertaintyAwareTrainer:
    def test_initialization(self, mock_model):
        trainer = UncertaintyAwareTrainer(
            weight=0.5,
            model=mock_model
        )
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert isinstance(trainer.ce_loss_fn, nn.CrossEntropyLoss)
        
    def test_split_dataloader(self, uncertainty_trainer, mock_data):
        split_loader = uncertainty_trainer.split_dataloader(mock_data)
        batch = next(iter(split_loader))
        assert len(batch) == 3  # (X, Y, Z)
        x, y, z = batch
        assert z.dtype == torch.long
        
    def test_calculate_loss(self, uncertainty_trainer):
        output = torch.randn(10, 3)
        target = torch.randint(0, 3, (10,))
        z_batch = torch.zeros(10)
        z_batch[5:] = 1
        
        # Test training loss
        loss = uncertainty_trainer.calculate_loss(
            output, target, z_batch, training=True)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar tensor
        
        # Test validation loss
        val_loss = uncertainty_trainer.calculate_loss(
            output, target, None, training=False)
        assert isinstance(val_loss, torch.Tensor)
        assert val_loss.ndim == 0
        
    def test_training_process(self, uncertainty_trainer, mock_data):
        # Train for a few epochs
        uncertainty_trainer.train(
            train_loader=mock_data,
            val_loader=mock_data,
            num_epochs=2
        )
        
    def test_validation_process(self, uncertainty_trainer, mock_data):
        val_loss = uncertainty_trainer.validate(mock_data)
        assert isinstance(val_loss, float)
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self, mock_model, mock_data):
        trainer = UncertaintyAwareTrainer(
            weight=0.5,
            model=mock_model,
            device=torch.device('cuda')
        )
        trainer.train(mock_data, num_epochs=2)
        
    def test_different_weights(self, mock_model, mock_data):
        # Test different weight values
        weights = [0.1, 1.0, 2.0]
        for weight in weights:
            trainer = UncertaintyAwareTrainer(
                weight=weight,
                model=mock_model
            )
            trainer.train(mock_data, num_epochs=1)