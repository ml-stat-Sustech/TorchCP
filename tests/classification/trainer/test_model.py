# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
import torch.nn as nn
from torch import Tensor


from torchcp.classification.trainer.model_zoo import TemperatureScalingModel,OrdinalClassifier, SurrogateCPModel

# Mock base model for testing
class MockBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2, bias=False)

    def forward(self, x):
        return torch.ones((x.shape[0], 10))  # Always return ones with 10 classes

# Fixtures
@pytest.fixture
def mock_base_model():
    return MockBaseModel()


@pytest.fixture
def mock_fc_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2, bias=False)

        def forward(self, x):
            return self.fc(x)

    return MockModel()


@pytest.fixture
def temp_scaling_model(mock_base_model):
    return TemperatureScalingModel(mock_base_model, temperature=2.0)


@pytest.fixture
def surrogate_cp_model(mock_fc_model):
    return SurrogateCPModel(mock_fc_model)


@pytest.fixture
def mock_classifier():
    return nn.Linear(10, 5)

@pytest.fixture
def ordinal_classifier(mock_classifier):
    return OrdinalClassifier(mock_classifier)

# Tests for TemperatureScalingModel
class TestTemperatureScalingModel:
    def test_initialization(self, temp_scaling_model):
        assert temp_scaling_model.get_temperature() == 2.0
        assert temp_scaling_model.is_base_model_frozen()

    def test_invalid_temperature_initialization(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            TemperatureScalingModel(MockBaseModel(), temperature=0.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            TemperatureScalingModel(MockBaseModel(), temperature=-1.0)

    def test_forward_pass(self, temp_scaling_model):
        input_tensor = torch.randn(5, 3, 224, 224)  # Batch of 5 images
        output = temp_scaling_model(input_tensor)
        
        # Check output shape
        assert output.shape == (5, 10)
        # Check if scaling is applied correctly (all values should be 0.5 since input is 1.0 and temp is 2.0)
        assert torch.allclose(output, torch.ones_like(output) * 0.5)

    def test_temperature_setter(self, temp_scaling_model):
        temp_scaling_model.set_temperature(3.0)
        assert temp_scaling_model.get_temperature() == 3.0

        with pytest.raises(ValueError, match="Temperature must be positive"):
            temp_scaling_model.set_temperature(0.0)

    def test_train_mode(self, temp_scaling_model):
        # Set to train mode
        temp_scaling_model.train()
        # Base model should still be in eval mode
        assert not temp_scaling_model.base_model.training
        # Temperature scaling model can be in train mode
        assert temp_scaling_model.training

        # Set to eval mode
        temp_scaling_model.eval()
        assert not temp_scaling_model.base_model.training
        assert not temp_scaling_model.training

# Tests for OrdinalClassifier
class TestOrdinalClassifier:
    def test_initialization(self, ordinal_classifier):
        assert ordinal_classifier.phi == "abs"
        assert ordinal_classifier.varphi == "abs"
        assert callable(ordinal_classifier.phi_function)
        assert callable(ordinal_classifier.varphi_function)

    def test_invalid_phi_varphi(self, mock_classifier):
        with pytest.raises(NotImplementedError):
            OrdinalClassifier(mock_classifier, phi="invalid")
        
        with pytest.raises(NotImplementedError):
            OrdinalClassifier(mock_classifier, varphi="invalid")

    def test_forward_pass(self, ordinal_classifier):
        input_tensor = torch.randn(3, 10)  # Batch of 3, input dim 10
        output = ordinal_classifier(input_tensor)
        
        # Check output shape matches input batch size and classifier output size
        assert output.shape == (3, 5)
        # Check output values are real numbers
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_small_input_dimension(self, ordinal_classifier):
        input_tensor = torch.randn(3, 2)  # Input dimension <= 2
        with pytest.raises(ValueError, match="The input dimension must be greater than 2"):
            ordinal_classifier(input_tensor)

    @pytest.mark.parametrize("phi,varphi", [
        ("abs", "abs"),
        ("square", "square"),
        ("abs", "square"),
        ("square", "abs")
    ])
    def test_different_transformations(self, mock_classifier, phi, varphi):
        model = OrdinalClassifier(mock_classifier, phi=phi, varphi=varphi)
        input_tensor = torch.randn(3, 10)
        output = model(input_tensor)
        
        assert output.shape == (3, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# Tests for SurrogateCPModel
class TestSurrogateCPModel:
    def test_initialization(self, mock_fc_model, surrogate_cp_model):
        assert surrogate_cp_model.base_model is mock_fc_model
        assert surrogate_cp_model.linear.in_features == mock_fc_model.fc.out_features
        assert surrogate_cp_model.linear.out_features == mock_fc_model.fc.out_features
        assert surrogate_cp_model.linear.bias is None
        assert surrogate_cp_model.is_base_model_frozen()

    def test_forward_pass(self, surrogate_cp_model):
        input_tensor = torch.randn(5, 10)
        output = surrogate_cp_model(input_tensor)
        
        assert output.shape == (5, 2)

    def test_train_mode(self, surrogate_cp_model):
        surrogate_cp_model.train()

        assert not surrogate_cp_model.base_model.training
        assert surrogate_cp_model.training

        surrogate_cp_model.eval()
        assert not surrogate_cp_model.base_model.training
        assert not surrogate_cp_model.training