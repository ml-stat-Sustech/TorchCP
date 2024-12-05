import pytest
import torch
from torchcp.classification.loss.cd import CDLoss
from torchcp.classification.predictor import SplitPredictor as Predictor
from torchcp.classification.score import THR

@pytest.fixture
def ds_instance():
    weight = 1.0
    predictor = Predictor(THR())
    epsilon = 1e-4
    return CDLoss(weight, predictor, epsilon)

def test_init(ds_instance):
    ds = ds_instance
    assert ds.weight == 1.0
    assert isinstance(ds.predictor, Predictor)
    assert ds.epsilon == 1e-4

def test_invalid_weight():
    with pytest.raises(ValueError):
        CDLoss(0, Predictor(THR()), 0.05)


def test_invalid_epsilon():
    with pytest.raises(ValueError):
        CDLoss(1.0, Predictor(THR()), 0)
        

def test_forward_with_different_epsilon():
    predictor = Predictor(THR())
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))
    epsilons = [1e-3, 1e-2, 1e-1]
    for epsilon in epsilons:
        ds_loss = CDLoss(1.0, predictor, epsilon)
        loss = ds_loss.forward(logits, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])

def test_forward_with_edge_cases():
    predictor = Predictor(THR())
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))
    
    # Test with very small epsilon
    ds_loss = CDLoss(1.0, predictor, 1e-10)
    loss = ds_loss.forward(logits, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    # Test with very large epsilon
    ds_loss = CDLoss(1.0, predictor, 1e+10)
    loss = ds_loss.forward(logits, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

