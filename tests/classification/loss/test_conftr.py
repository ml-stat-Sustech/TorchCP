import pytest
import torch
from torchcp.classification.loss.conftr import ConfTr
from torchcp.classification.predictors import SplitPredictor as Predictor
from torchcp.classification.scores import THR


@pytest.fixture
def conftr_instance():
    weight = 1.0
    predictor = Predictor(THR())
    alpha = 0.05
    fraction = 0.2
    loss_type = "valid"
    target_size = 1
    loss_transform = "square"
    epsilon = 1e-4
    return ConfTr(weight, predictor, alpha, fraction, True, epsilon, loss_type, target_size, loss_transform)

def test_init(conftr_instance):
    conftr = conftr_instance
    assert conftr.weight == 1.0
    assert isinstance(conftr.predictor, Predictor)
    assert conftr.alpha == 0.05
    assert conftr.fraction == 0.2
    assert conftr.loss_type == "valid"
    assert conftr.target_size == 1
    assert conftr.epsilon == 1e-4
    assert conftr.transform == torch.square

def test_invalid_fraction():
    with pytest.raises(ValueError):
        ConfTr(1.0, Predictor(THR()), 0.05, 0)

def test_invalid_loss_type():
    with pytest.raises(ValueError):
        ConfTr(1.0, Predictor(THR()), 0.05, 0.2, loss_type="invalid")

def test_invalid_target_size():
    with pytest.raises(ValueError):
        ConfTr(1.0, Predictor(THR()), 0.05, 0.2, target_size=2)

def test_invalid_loss_transform():
    with pytest.raises(ValueError):
        ConfTr(1.0, Predictor(THR()), 0.05, 0.2, loss_transform="invalid")

def test_invalid_epsilon():
    with pytest.raises(ValueError):
        ConfTr(1.0, Predictor(THR()), 0.05, 0.2, epsilon=0)

def test_compute_loss(conftr_instance):
    conftr = conftr_instance
    test_scores = torch.randn(10, 5)
    test_labels = torch.randint(0, 2, (10,))
    tau = 0.5
    loss = conftr.compute_loss(test_scores, test_labels, tau)
    assert isinstance(loss, torch.Tensor)

def test_compute_hinge_size_loss(conftr_instance):
    conftr = conftr_instance
    pred_sets = torch.sigmoid(torch.randn(10, 5))
    labels = torch.randint(0, 2, (10,))
    loss = conftr._ConfTr__compute_hinge_size_loss(pred_sets, labels)
    assert isinstance(loss, torch.Tensor)

def test_compute_probabilistic_size_loss(conftr_instance):
    conftr = conftr_instance
    pred_sets = torch.sigmoid(torch.randn(5, 10))
    labels = torch.randint(0, 2, (5,))
    loss = conftr._ConfTr__compute_probabilistic_size_loss(pred_sets, labels)
    assert isinstance(loss, torch.Tensor)

def test_compute_coverage_loss(conftr_instance):
    conftr = conftr_instance
    pred_sets = torch.sigmoid(torch.randn(10, 5))
    labels = torch.randint(0, 5, (10,))
    loss = conftr._ConfTr__compute_coverage_loss(pred_sets, labels)
    assert isinstance(loss, torch.Tensor)

def test_compute_classification_loss(conftr_instance):
    conftr = conftr_instance
    pred_sets = torch.sigmoid(torch.randn(10, 5))
    labels = torch.randint(0, 5, (10,))
    loss = conftr._ConfTr__compute_classification_loss(pred_sets, labels)
    assert isinstance(loss, torch.Tensor)
    
def test_loss_transform_square():
    conftr = ConfTr(1.0, Predictor(THR()), 0.05, 0.2, loss_transform="square")
    assert conftr.transform == torch.square

def test_loss_transform_abs():
    conftr = ConfTr(1.0, Predictor(THR()), 0.05, 0.2, loss_transform="abs")
    assert conftr.transform == torch.abs

def test_loss_transform_log():
    conftr = ConfTr(1.0, Predictor(THR()), 0.05, 0.2, loss_transform="log")
    assert conftr.transform == torch.log