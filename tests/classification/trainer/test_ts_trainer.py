import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from typing import List

from torchcp.classification.trainer.ts_trainer import TSTrainer
from torchcp.classification.trainer.model import TemperatureScalingModel

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)
        
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def base_model():
    return SimpleModel()

@pytest.fixture
def optimizer(base_model):
    return SGD([{'params': base_model.parameters()}], lr=0.01)

@pytest.fixture
def loss_fn():
    return nn.CrossEntropyLoss()

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def ts_trainer(base_model, optimizer, loss_fn, device):
    return TSTrainer(
        model=base_model,
        temperature=1.5,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        verbose=False
    )

def test_initialization(base_model, optimizer, loss_fn, device):
    # Test normal initialization
    trainer = TSTrainer(
        model=base_model,
        temperature=1.5,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )
    
    assert isinstance(trainer.model, TemperatureScalingModel)
    assert trainer.model.get_temperature() == pytest.approx(1.5)
    assert trainer.device == device
    
    # Test with multiple loss functions and weights
    loss_fns = [nn.CrossEntropyLoss(), nn.MSELoss()]
    loss_weights = [0.7, 0.3]
    
    trainer = TSTrainer(
        model=base_model,
        temperature=1.5,
        optimizer=optimizer,
        loss_fn=loss_fns,
        loss_weights=loss_weights,
        device=device
    )
    
    assert isinstance(trainer.loss_fn, List)
    assert len(trainer.loss_fn) == 2

def test_invalid_initialization(base_model, optimizer, loss_fn, device):
    # Test invalid temperature
    with pytest.raises(ValueError):
        TSTrainer(
            model=base_model,
            temperature=-1.0,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )
    
    # Test invalid loss weights
    with pytest.raises(ValueError):
        TSTrainer(
            model=base_model,
            temperature=1.5,
            optimizer=optimizer,
            loss_fn=[loss_fn, loss_fn],
            loss_weights=[0.5],  # Wrong number of weights
            device=device
        )

def test_model_wrapping(ts_trainer):
    # Check if base model is properly wrapped
    assert isinstance(ts_trainer.model, TemperatureScalingModel)
    assert ts_trainer.model.is_base_model_frozen()
    assert not ts_trainer.model.base_model.training

def test_device_handling(ts_trainer, device):
    # Create sample input
    x = torch.randn(5, 10)
    
    # Move input to device
    x = x.to(device)
    
    model_device = next(ts_trainer.model.parameters()).device
    assert model_device.type == device.type, "Model should be on same device type as specified"
    
    # Test forward pass with device
    output = ts_trainer.model(x)
    assert output.device.type == device.type, "Output should be on same device type as input"

def test_training_mode(ts_trainer):
    # Test training mode switches
    ts_trainer.model.train()
    assert not ts_trainer.model.base_model.training  # Base model should stay in eval mode
    
    ts_trainer.model.eval()
    assert not ts_trainer.model.base_model.training

def test_optimizer_setup(ts_trainer):
    # Check if optimizer is properly set up
    assert ts_trainer.optimizer is not None
    
    # Only temperature parameter should be optimized
    optimized_params = []
    for param_group in ts_trainer.optimizer.param_groups:
        optimized_params.extend([p for p in param_group['params'] if p.requires_grad])
    
    assert len(optimized_params) == 1  # Only temperature parameter
    assert torch.equal(optimized_params[0], ts_trainer.model.temperature)

def test_loss_function_handling(base_model, optimizer, device):
    # Test single loss function
    single_loss = nn.CrossEntropyLoss()
    trainer = TSTrainer(
        model=base_model,
        temperature=1.5,
        optimizer=optimizer,
        loss_fn=single_loss,
        device=device
    )
    assert callable(trainer.loss_fn)
    
    # Test multiple loss functions
    multi_loss = [nn.CrossEntropyLoss(), nn.MSELoss()]
    weights = [0.6, 0.4]
    trainer = TSTrainer(
        model=base_model,
        temperature=1.5,
        optimizer=optimizer,
        loss_fn=multi_loss,
        loss_weights=weights,
        device=device
    )
    assert isinstance(trainer.loss_fn, List)
    assert len(trainer.loss_fn) == 2
    assert torch.equal(trainer.loss_weights, torch.tensor(weights, device=device))

def test_forward_pass(ts_trainer, device):
    # Create sample input
    x = torch.randn(5, 10).to(device)
    
    # Test forward pass
    output = ts_trainer.model(x)
    
    # Check output properties
    assert output.shape == (5, 3)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    # Test temperature scaling effect
    initial_output = output.clone()
    ts_trainer.model.set_temperature(3.0)
    scaled_output = ts_trainer.model(x)
    
    # Check scaling relationship
    ratio = (initial_output / scaled_output).mean().item()
    assert ratio == pytest.approx(3.0/1.5, rel=1e-5)