import pytest
import torch
import torch.nn as nn

from torchcp.classification.predictor.base import BasePredictor
from torchcp.classification.utils import ConfCalibrator


class DummyModel(nn.Module):
    """A simple model for testing"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


class ConcretePredictor(BasePredictor):
    """Concrete implementation of BasePredictor for testing"""

    def calibrate(self, cal_dataloader, alpha):
        pass

    def predict(self, x_batch):
        pass


@pytest.fixture
def dummy_score_function():
    """Score function fixture"""

    def score_fn(logits, labels):
        return 1 - logits[torch.arange(len(labels)), labels]

    return score_fn


def test_base_predictor_initialization():
    """Test BasePredictor initialization with various parameters"""
    score_fn = lambda x, y: x
    model = DummyModel()

    # Test successful initialization
    predictor = ConcretePredictor(score_fn, model)
    assert predictor.score_function == score_fn
    assert predictor._model == model
    assert predictor._model.training == False  # Should be in eval mode
    assert isinstance(predictor._logits_transformation, ConfCalibrator.registry_ConfCalibrator("TS"))

    # Test initialization without model
    predictor = ConcretePredictor(score_fn)
    assert predictor._model is None

    # Test invalid temperature
    with pytest.raises(ValueError, match="temperature must be greater than 0"):
        ConcretePredictor(score_fn, model, temperature=0)

    with pytest.raises(ValueError, match="temperature must be greater than 0"):
        ConcretePredictor(score_fn, model, temperature=-1)


def test_device_handling():
    """Test device handling functionality"""
    score_fn = lambda x, y: x
    model = DummyModel()

    # Test CPU device
    predictor = ConcretePredictor(score_fn, model)
    assert predictor.get_device() == torch.device('cpu')

    # Test device with no model
    predictor = ConcretePredictor(score_fn)
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        cuda_idx = torch.cuda.current_device()
        device = torch.device(f"cuda:{cuda_idx}")
    assert predictor.get_device() == device

    # Test CUDA device if available
    if torch.cuda.is_available():
        model = model.cuda()
        predictor = ConcretePredictor(score_fn, model)
        assert predictor.get_device() == torch.device('cuda:0')


def test_generate_prediction_set():
    """Test _generate_prediction_set method"""
    score_fn = lambda x, y: x
    predictor = ConcretePredictor(score_fn)

    # Test with simple scores and threshold
    scores = torch.tensor([0.1, 0.5, 0.8, 0.3, 0.9])
    q_hat = torch.tensor(0.6)
    expected = torch.tensor([1, 1, 0, 1, 0])

    result = predictor._generate_prediction_set(scores, q_hat)
    assert torch.equal(result, expected)

    # Test with different threshold
    q_hat = torch.tensor(0.2)
    expected = torch.tensor([1, 0, 0, 0, 0])
    result = predictor._generate_prediction_set(scores, q_hat)
    assert torch.equal(result, expected)

    # Test with all scores below threshold
    q_hat = torch.tensor(1.0)
    expected = torch.ones_like(scores)
    result = predictor._generate_prediction_set(scores, q_hat)
    assert torch.equal(result, expected)

    # Test with all scores above threshold
    q_hat = torch.tensor(0.0)
    expected = torch.zeros_like(scores)
    result = predictor._generate_prediction_set(scores, q_hat)
    assert torch.equal(result, expected)


def test_abstract_methods():
    """Test abstract methods raise NotImplementedError"""
    with pytest.raises(NotImplementedError):
        BasePredictor(lambda x, y: x).calibrate(None, 0.1)

    with pytest.raises(NotImplementedError):
        BasePredictor(lambda x, y: x).predict(torch.randn(10, 5))


def test_logits_transformation():
    """Test logits transformation with different temperatures"""
    score_fn = lambda x, y: x
    model = DummyModel()

    # Test with default temperature (1.0)
    predictor = ConcretePredictor(score_fn, model)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    transformed = predictor._logits_transformation(logits)
    assert torch.allclose(transformed, logits)

    # Test with different temperature
    predictor = ConcretePredictor(score_fn, model, temperature=2.0)
    transformed = predictor._logits_transformation(logits)
    expected = logits / 2.0
    assert torch.allclose(transformed, expected)


@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_prediction_set_shapes(batch_size):
    """Test prediction set generation with different batch sizes"""
    score_fn = lambda x, y: x
    predictor = ConcretePredictor(score_fn)

    scores = torch.randn(batch_size, 10)  # 10 classes
    q_hat = torch.tensor(0.5)

    result = predictor._generate_prediction_set(scores, q_hat)
    assert result.shape == scores.shape
    assert result.dtype == torch.int
    assert torch.all((result == 0) | (result == 1))
