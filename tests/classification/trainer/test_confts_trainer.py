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

from torchcp.classification.loss import ConfTS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS
from torchcp.classification.trainer import ConfTSTrainer, TSTrainer
from torchcp.classification.trainer import TemperatureScalingModel


@pytest.fixture
def mock_model():
    return nn.Linear(10, 2)  # Simple model for testing


@pytest.fixture
def mock_optimizer(mock_model):
    return optim.SGD(mock_model.parameters(), lr=0.01)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_confts_trainer_init_default(mock_model, mock_optimizer, device):
    """Test ConfTSTrainer initialization with default parameters"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda:0" if device == "cuda" else "cpu")

    temperature = 1.0
    trainer = ConfTSTrainer(
        model=mock_model,
        temperature=temperature,
        optimizer=mock_optimizer,
        device=device,
    )

    assert trainer.device == device
    assert trainer.verbose == True
    assert isinstance(trainer.loss_fn, ConfTS)
    assert isinstance(trainer.loss_fn.predictor, SplitPredictor)
    assert isinstance(trainer.loss_fn.predictor.score_function, APS)
    assert trainer.loss_fn.fraction == 0.5


def test_confts_trainer_init_custom_device(mock_model, mock_optimizer):
    """Test ConfTSTrainer initialization with custom device"""
    temperature = 1.0
    device = torch.device('cpu')
    trainer = ConfTSTrainer(
        model=mock_model,
        temperature=temperature,
        optimizer=mock_optimizer,
        device=device,
        verbose=False
    )

    assert trainer.device == device
    assert trainer.verbose == False


def test_confts_trainer_inheritance(mock_model, mock_optimizer):
    """Test if ConfTSTrainer properly inherits from TSTrainer"""

    trainer = ConfTSTrainer(
        model=mock_model,
        temperature=1.0,
        optimizer=mock_optimizer
    )

    assert isinstance(trainer, TSTrainer)


def test_confts_trainer_model_wrapper(mock_model, mock_optimizer):
    """Test if model is properly wrapped in TemperatureScalingModel"""

    trainer = ConfTSTrainer(
        model=mock_model,
        temperature=1.0,
        optimizer=mock_optimizer
    )

    assert isinstance(trainer.model, TemperatureScalingModel)


@pytest.fixture
def synthetic_data():
    # Create small synthetic dataset
    X = torch.randn(120, 10)  # 100 samples, 10 features
    y = torch.randint(0, 10, (120,))  # Binary classification
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=30)
    val_loader = DataLoader(dataset, batch_size=30)
    return train_loader, val_loader


@pytest.fixture
def trainer_setup():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU())
    optimizer = torch.optim.Adam(model.parameters())
    temperature = 1.0
    trainer = ConfTSTrainer(
        model=model,
        temperature=temperature,
        optimizer=optimizer,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    return trainer


def test_train_without_validation(trainer_setup, synthetic_data):
    train_loader, _ = synthetic_data
    num_epochs = 2

    trainer_setup.train(
        train_loader=train_loader,
        num_epochs=num_epochs
    )

    # Test model is in training mode after training
    assert trainer_setup.model.training


def test_train_with_validation(trainer_setup, synthetic_data, tmp_path):
    train_loader, val_loader = synthetic_data
    num_epochs = 2
    save_path = os.path.join(tmp_path, "best_model.pt")

    trainer_setup.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_path=save_path
    )

    # Test model file was saved
    assert os.path.exists(save_path)


def test_train_metrics_tracking(trainer_setup, synthetic_data):
    train_loader, val_loader = synthetic_data
    num_epochs = 2

    # Train with validation to get metrics
    trainer_setup.verbose = False  # Disable logging for test
    trainer_setup.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )

    # Verify model has expected attributes after training
    assert hasattr(trainer_setup.model, 'temperature')
    assert isinstance(trainer_setup.loss_fn, ConfTS)
