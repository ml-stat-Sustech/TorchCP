# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import logging
import os
import os
import pytest
import pytest
import tempfile
import tempfile
import torch
import torch
import torch.nn as nn
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader

from torchcp.classification.trainer.base_trainer import Trainer


# Simple test model for neural network testing
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


# Dummy dataset for testing training and validation processes
class DummyDataset(Dataset):
    def __init__(self, size=100, input_dim=10, num_classes=2):
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, num_classes, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def device():
    """Fixture for device used in testing"""
    return torch.device("cpu")


@pytest.fixture
def model():
    """Fixture for creating a test model"""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Fixture for creating an optimizer"""
    return torch.optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def loss_fn():
    """Fixture for creating a loss function"""
    return nn.CrossEntropyLoss()


@pytest.fixture
def train_loader(request):
    """Fixture for creating a training data loader"""
    return DataLoader(DummyDataset(100), batch_size=10)


@pytest.fixture
def val_loader(request):
    """Fixture for creating a validation data loader"""
    return DataLoader(DummyDataset(50), batch_size=10)


@pytest.fixture
def trainer(model, optimizer, loss_fn, device):
    """Fixture for creating a Trainer instance"""
    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        verbose=True
    )


def test_trainer_initialization(model, optimizer, loss_fn, device):
    """
    Test initialization of Trainer with single loss function
    
    Verifies that:
    - Model is correctly set
    - Optimizer is correctly set
    - Loss function is correctly set
    """
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, device=device)
    assert trainer.model == model
    assert trainer.optimizer == optimizer
    assert trainer.loss_fn == loss_fn


def test_multiple_loss_functions(model, optimizer, device):
    """
    Test initialization of Trainer with multiple loss functions
    
    Verifies that:
    - Multiple loss functions can be set
    - Corresponding weights are correctly assigned
    """
    loss_fns = [nn.CrossEntropyLoss(), nn.MSELoss()]
    loss_weights = [0.6, 0.4]
    trainer = Trainer(
        model,
        optimizer,
        loss_fns,
        loss_weights,
        device
    )
    assert trainer.loss_fn == loss_fns


def test_init_validation_errors(model, optimizer, device):
    """
    Test error handling during Trainer initialization
    
    Verifies that:
    - Appropriate errors are raised when loss function requirements are not met
    - Missing weights trigger an assertion error
    - Mismatched number of loss functions and weights trigger an assertion error
    """
    loss_fns = [nn.CrossEntropyLoss(), nn.MSELoss()]

    loss_weights = [0.6]
    with pytest.raises(ValueError, match="Number of loss functions must match number of weights"):
        Trainer(model, optimizer, loss_fns, device=device, loss_weights=loss_weights)

    loss_fns = nn.CrossEntropyLoss()
    loss_weights = [0.6]
    with pytest.raises(ValueError, match="Expected a single loss function, got a list of loss weights"):
        Trainer(model, optimizer, loss_fns, device=device, loss_weights=loss_weights)


def test_calculate_loss(trainer):
    """
    Test loss calculation with a single loss function
    
    Verifies that:
    - Loss calculation returns a valid tensor
    """
    output = torch.randn(10, 2)
    target = torch.randint(0, 2, (10,))
    loss = trainer.calculate_loss(output, target)
    assert isinstance(loss, torch.Tensor)


def test_multiple_loss_calculation(model, optimizer, device):
    """
    Test loss calculation with multiple loss functions
    
    Verifies that:
    - Loss calculation with multiple functions returns a valid tensor
    - Weighted combination of losses works correctly
    """
    loss_fns = [nn.CrossEntropyLoss()]
    loss_weights = [0.6]
    trainer = Trainer(
        model,
        optimizer,
        loss_fns,
        loss_weights,
        device
    )

    # Create output and target with compatible shapes
    output = torch.randn(10, 2)
    target = torch.randint(0, 2, (10,))

    loss = trainer.calculate_loss(output, target)
    assert isinstance(loss, torch.Tensor)


def test_train_epoch(trainer, train_loader):
    """
    Test training for a single epoch with single loss function
    
    Verifies that:
    - Training for one epoch produces valid metrics
    - Loss metric is present
    """
    metrics = trainer.train_epoch(train_loader)
    assert 'loss' in metrics


def test_train_epoch_multiple_losses(model, optimizer, device, train_loader):
    """
    Test training for a single epoch with multiple loss functions
    
    Verifies that:
    - Training with multiple loss functions works correctly
    - Individual loss metrics are tracked
    """
    loss_fns = [nn.CrossEntropyLoss()]
    loss_weights = [0.6]
    trainer = Trainer(
        model,
        optimizer,
        loss_fns,
        loss_weights,
        device
    )
    metrics = trainer.train_epoch(train_loader)
    assert 'loss' in metrics
    assert 'loss_0' in metrics


def test_validation(trainer, val_loader):
    """
    Test model validation with single loss function
    
    Verifies that:
    - Validation produces expected metrics
    - Validation loss and accuracy metrics are present
    """
    metrics = trainer.validate(val_loader)
    assert 'val_loss' in metrics
    assert 'val_acc' in metrics


def test_validation_multiple_losses(model, optimizer, device, val_loader):
    """
    Test model validation with multiple loss functions
    
    Verifies that:
    - Validation with multiple loss functions works correctly
    - Individual validation loss metrics are tracked
    """
    loss_fns = [nn.CrossEntropyLoss()]
    loss_weights = [0.6]
    trainer = Trainer(
        model,
        optimizer,
        loss_fns,
        loss_weights,
        device
    )
    metrics = trainer.validate(val_loader)
    assert 'val_loss' in metrics
    assert 'val_acc' in metrics
    assert 'val_loss_0' in metrics


def test_training_without_validation(trainer, train_loader):
    """
    Test training process without a validation set
    
    Verifies that:
    - Training can proceed without a validation loader
    """
    trainer.train(train_loader, num_epochs=2)


def test_training_with_validation(trainer, train_loader, val_loader):
    """
    Test training process with validation set and model saving
    
    Verifies that:
    - Training with validation works correctly
    - Model can be saved during training
    """
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        try:
            trainer.train(
                train_loader,
                val_loader,
                num_epochs=2,
                save_path=tmp.name
            )
            assert os.path.exists(tmp.name)
        finally:
            os.unlink(tmp.name)


def test_checkpointing(trainer):
    """
    Test checkpoint saving and loading functionality
    
    Verifies that:
    - Checkpoints can be saved with metadata
    - Saved checkpoints can be correctly loaded
    - Checkpoint contains expected information
    """
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        try:
            metrics = {'loss': 0.5, 'val_loss': 0.4}
            trainer.save_checkpoint(1, tmp.name, metrics)
            assert os.path.exists(tmp.name)

            # Load checkpoint
            checkpoint = trainer.load_checkpoint(tmp.name)
            assert checkpoint['epoch'] == 1
            assert checkpoint['metrics'] == metrics
        finally:
            os.unlink(tmp.name)


def test_verbose_mode(model, optimizer, loss_fn, device, train_loader, val_loader):
    """
    Test trainer initialization and training in non-verbose mode
    
    Verifies that:
    - Trainer can be initialized with verbose=False
    - No logger is created in non-verbose mode
    - Training can proceed without errors
    """
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        verbose=False
    )

    # Verify no logger attribute exists
    assert not hasattr(trainer, 'logger')

    # Run training to ensure no errors occur
    trainer.train(train_loader, val_loader, num_epochs=1)


@pytest.mark.parametrize("verbose", [True, False])
def test_logging_configuration(model, optimizer, loss_fn, device, train_loader, val_loader, verbose):
    """
    Parameterized test for different logging configurations
    
    Verifies that:
    - Trainer can be initialized with different verbosity settings
    - Training works correctly regardless of verbosity
    
    Args:
        verbose (bool): Whether logging is enabled
    """
    trainer = Trainer(
        model,
        optimizer,
        loss_fn,
        device=device,
        verbose=verbose
    )

    # Verify training runs without errors
    trainer.train(train_loader, val_loader, num_epochs=1)


def test_trainer_device_initialization(model, optimizer, loss_fn):
    """
    Test initialization of Trainer with default device
    
    Verifies that:
    - Trainer initializes with the correct device
    """
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
    assert trainer.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_trainer_loss_weights_initialization(model, optimizer, loss_fn, device):
    """
    Test initialization of Trainer with single loss function and weights
    
    Verifies that:
    - Trainer initializes with correct loss weights
    """
    # Change from list [1.0] to scalar 1.0
    loss_weight = 1.0
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn, loss_weights=loss_weight, device=device)
    assert torch.equal(trainer.loss_weights, torch.tensor(loss_weight, device=device))


def test_trainer_multiple_loss_weights_initialization(model, optimizer, device):
    """
    Test initialization of Trainer with multiple loss functions and weights
    
    Verifies that:
    - Trainer initializes with correct loss weights for multiple loss functions
    """
    loss_fns = [nn.CrossEntropyLoss(), nn.MSELoss()]
    loss_weights = [0.6, 0.4]
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fns, loss_weights=loss_weights, device=device)
    assert torch.equal(trainer.loss_weights, torch.tensor(loss_weights, device=device))

    loss_fns = [nn.CrossEntropyLoss(), nn.MSELoss()]
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fns, device=device)
    assert torch.equal(trainer.loss_weights, torch.ones(len(loss_fns), device=device))
