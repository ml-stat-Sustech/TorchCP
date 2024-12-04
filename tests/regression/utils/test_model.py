import pytest
import torch
import torch.nn as nn
from torchcp.regression.utils.model import NonLinearNet, Softmax, build_regression_model


@pytest.fixture
def mock_input():
    # Create a mock input tensor for testing forward pass
    return torch.randn(10, 5)  # batch_size=10, in_shape=5


def test_non_linear_net_initialization():
    # Test initialization of NonLinearNet class
    model = NonLinearNet(input_dim=5, output_dim=2, hidden_size=10, dropout=0.5)

    # Check attributes
    assert model.hidden_size == 10, f"Expected hidden_size 10, got {model.hidden_size}"
    assert model.input_dim == 5, f"Expected in_shape 5, got {model.in_shape}"
    assert model.output_dim == 2, f"Expected out_shape 2, got {model.out_shape}"
    assert model.dropout == 0.5, f"Expected dropout 0.5, got {model.dropout}"

    # Check that base_model is a Sequential container with the correct layers
    assert isinstance(model.base_model, nn.Sequential), f"Expected Sequential, got {type(model.base_model)}"
    assert len(model.base_model) == 7, f"Expected 7 layers, got {len(model.base_model)}"
    assert isinstance(model.base_model[0], nn.Linear), f"Expected first layer to be nn.Linear, got {type(model.base_model[0])}"
    assert isinstance(model.base_model[1], nn.ReLU), f"Expected second layer to be nn.ReLU, got {type(model.base_model[1])}"


def test_non_linear_net_forward(mock_input):
    # Test forward pass of NonLinearNet
    model = NonLinearNet(input_dim=5, output_dim=2, hidden_size=10, dropout=0.5)
    output = model(mock_input)

    # Check output shape
    assert output.shape == (mock_input.shape[0], 2), f"Expected output shape (10, 2), got {output.shape}"


def test_softmax_initialization():
    # Test initialization of Softmax class
    model = Softmax(input_dim=5, output_dim=2, hidden_size=10, dropout=0.5)

    # Check that Softmax contains an instance of NonLinearNet and a Softmax layer
    assert isinstance(model.base_model[0], NonLinearNet), "Expected base_model[0] to be an instance of NonLinearNet"
    assert isinstance(model.base_model[1], nn.Softmax), "Expected base_model[1] to be nn.Softmax"


def test_softmax_forward(mock_input):
    # Test forward pass of Softmax
    model = Softmax(input_dim=5, output_dim=2, hidden_size=10, dropout=0.5)
    output = model(mock_input)

    # Check output shape
    assert output.shape == (mock_input.shape[0], 2), f"Expected output shape (10, 2), got {output.shape}"

    # Check that output is a probability distribution (sum of softmax output should be 1 for each sample)
    assert torch.allclose(output.sum(dim=1), torch.ones(mock_input.shape[0]), atol=1e-6), "Softmax output doesn't sum to 1"


def test_build_regression_model():
    # Test if build_regression_model returns the correct class
    model_class = build_regression_model("NonLinearNet")
    assert model_class == NonLinearNet, f"Expected NonLinearNet, got {model_class}"

    model_class = build_regression_model("NonLinearNet_with_Softmax")
    assert model_class == Softmax, f"Expected Softmax, got {model_class}"

    # Test for invalid model_name
    with pytest.raises(NotImplementedError):
        build_regression_model("InvalidModelName")
