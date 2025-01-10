# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
import torch.nn as nn

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
def temp_model(base_model):
    return TemperatureScalingModel(base_model, temperature=1.5)


def test_initialization():
    # Test normal initialization
    base_model = SimpleModel()
    temp_model = TemperatureScalingModel(base_model, temperature=1.5)
    assert temp_model.get_temperature() == pytest.approx(1.5)
    assert temp_model.is_base_model_frozen()

    # Test invalid temperature
    with pytest.raises(ValueError):
        TemperatureScalingModel(base_model, temperature=0.0)
    with pytest.raises(ValueError):
        TemperatureScalingModel(base_model, temperature=-1.0)


def test_base_model_frozen(temp_model):
    # Check if base model parameters are frozen
    assert temp_model.is_base_model_frozen()

    for param in temp_model.base_model.parameters():
        assert not param.requires_grad

    # Check if temperature parameter is trainable
    assert temp_model.temperature.requires_grad


def test_forward_pass(temp_model):
    # Test forward pass with random input
    batch_size = 5
    input_size = 10
    x = torch.randn(batch_size, input_size)

    output = temp_model(x)

    # Check output shape
    assert output.shape == (batch_size, 3)

    # Check if output is scaled by temperature
    with torch.no_grad():
        base_output = temp_model.base_model(x)
        expected_output = base_output / temp_model.get_temperature()
        assert torch.allclose(output, expected_output)


def test_temperature_getter_setter(temp_model):
    # Test get_temperature
    assert temp_model.get_temperature() == pytest.approx(1.5)

    # Test set_temperature with valid value
    temp_model.set_temperature(2.0)
    assert temp_model.get_temperature() == pytest.approx(2.0)

    # Test set_temperature with invalid values
    with pytest.raises(ValueError):
        temp_model.set_temperature(0.0)
    with pytest.raises(ValueError):
        temp_model.set_temperature(-1.0)


def test_gradient_flow(temp_model):
    # Create sample input and target
    x = torch.randn(5, 10)
    target = torch.randint(0, 3, (5,))

    # Forward pass
    output = temp_model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    loss.backward()

    # Check gradients
    assert temp_model.temperature.grad is not None

    # Base model parameters should have no gradients
    for param in temp_model.base_model.parameters():
        assert param.grad is None


def test_model_eval_mode(temp_model):
    # Check if base model is in eval mode
    assert not temp_model.base_model.training

    # Switching to train mode should not affect base model
    temp_model.train()
    assert not temp_model.base_model.training

    # Switching back to eval mode
    temp_model.eval()
    assert not temp_model.base_model.training


def test_temperature_persistence(base_model):
    # Create model and modify temperature
    temp_model = TemperatureScalingModel(base_model, temperature=1.0)
    temp_model.set_temperature(2.0)

    # Save and load state dict
    state_dict = temp_model.state_dict()

    # Create new model and load state dict
    new_temp_model = TemperatureScalingModel(base_model, temperature=1.0)
    new_temp_model.load_state_dict(state_dict)

    # Check if temperature was correctly loaded
    assert new_temp_model.get_temperature() == pytest.approx(2.0)


def test_output_scaling(temp_model):
    # Test if output scaling is correct for different temperature values
    x = torch.randn(5, 10)

    # Get output with temperature = 1.5 (initial value)
    output1 = temp_model(x)

    # Change temperature to 3.0
    temp_model.set_temperature(3.0)
    output2 = temp_model(x)

    # Check if scaling relationship holds
    expected_ratio = 3.0 / 1.5
    actual_ratio = (output1 / output2).mean().item()
    assert actual_ratio == pytest.approx(expected_ratio, rel=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_consistency(temp_model, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0" if device == "cuda" else "cpu")
    # Move model to device
    temp_model = temp_model.to(device)

    # Create input on same device
    x = torch.randn(4, 10).to(device)

    # Forward pass
    output = temp_model(x)

    # Check device consistency
    assert output.device == torch.device(device)
    assert temp_model.temperature.device == torch.device(device)
    assert next(temp_model.base_model.parameters()).device == torch.device(device)
