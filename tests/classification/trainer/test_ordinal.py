import pytest
import torch
import torch.nn as nn
from torchcp.classification.trainer import OrdinalClassifier

class MockClassifier(nn.Module):
    def forward(self, x):
        return x

@pytest.fixture
def ordinal_classifier_instance():
    classifier = MockClassifier()
    return OrdinalClassifier(classifier, phi="abs", varphi="abs")

def test_init(ordinal_classifier_instance):
    model = ordinal_classifier_instance
    assert isinstance(model.classifier, MockClassifier)
    assert model.phi == "abs"
    assert model.varphi == "abs"

def test_invalid_phi():
    classifier = MockClassifier()
    with pytest.raises(NotImplementedError):
        OrdinalClassifier(classifier, phi="invalid", varphi="abs")

def test_invalid_varphi():
    classifier = MockClassifier()
    with pytest.raises(NotImplementedError):
        OrdinalClassifier(classifier, phi="abs", varphi="invalid")

def test_forward(ordinal_classifier_instance):
    model = ordinal_classifier_instance
    x = torch.randn(3, 10)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape

def test_forward_invalid_input_dimension(ordinal_classifier_instance):
    model = ordinal_classifier_instance
    x = torch.randn(3, 2)
    with pytest.raises(ValueError):
        model(x)

def test_forward_with_square_phi_varphi():
    classifier = MockClassifier()
    model = OrdinalClassifier(classifier, phi="square", varphi="square")
    x = torch.randn(3, 10)
    output = model(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape

if __name__ == '__main__':
    pytest.main()