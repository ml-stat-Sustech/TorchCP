import pytest
import torch
import torch.nn as nn
from torchcp.classification.predictors.split import SplitPredictor as Split

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)
    
    def forward(self, x):
        return self.linear(x)

def mock_score_function(logits, labels):
    return torch.abs(logits - labels.unsqueeze(1))

@pytest.fixture
def split_instance():
    model = MockModel()
    return Split(score_function=mock_score_function, model=model)

@pytest.fixture
def calibration_data():
    x = torch.randn(10, 2)
    y = torch.randint(0, 3, (10,))
    return [(x, y)]

def test_split_initialization():
    # Test basic initialization
    split = Split(score_function=mock_score_function)
    assert split.score_function == mock_score_function
    assert split._model is None

    # Test with model and custom temperature
    model = MockModel()
    split = Split(score_function=mock_score_function, model=model, temperature=0.5)
    assert split._model is model

def test_calibration(split_instance, calibration_data):
    # Test valid calibration
    split_instance.calibrate(calibration_data, alpha=0.1)
    assert hasattr(split_instance, 'q_hat')
    assert isinstance(split_instance.q_hat, torch.Tensor)

    # Test invalid alpha values
    with pytest.raises(ValueError, match="alpha should be a value in"):
        split_instance.calibrate(calibration_data, alpha=0)
    with pytest.raises(ValueError, match="alpha should be a value in"):
        split_instance.calibrate(calibration_data, alpha=1)

def test_missing_model():
    split = Split(score_function=mock_score_function)
    x = torch.randn(5, 2)
    
    # Test calibration without model
    with pytest.raises(ValueError, match="Model is not defined"):
        split.calibrate([(x, torch.zeros(5))], alpha=0.1)
    
    # Test prediction without model
    with pytest.raises(ValueError, match="Model is not defined"):
        split.predict(x)

def test_prediction(split_instance):
    # Calibrate first
    x = torch.randn(10, 2)
    y = torch.randint(0, 3, (10,))
    split_instance.calibrate([(x, y)], alpha=0.1)
    
    # Test prediction
    x_test = torch.randn(5, 2)
    pred_sets = split_instance.predict(x_test)
    assert isinstance(pred_sets, list)
    assert len(pred_sets) == 5

def test_calculate_threshold(split_instance):
    logits = torch.randn(10, 3)
    labels = torch.randint(0, 3, (10,))
    
    split_instance.calculate_threshold(logits, labels, alpha=0.1)
    assert hasattr(split_instance, 'q_hat')
    assert isinstance(split_instance.q_hat, torch.Tensor)

def test_device_handling(split_instance):
    if torch.cuda.is_available():
        x = torch.randn(5, 2).cuda()
        y = torch.randint(0, 3, (5,)).cuda()
        
        # Test calibration on GPU
        split_instance.calibrate([(x, y)], alpha=0.1)
        assert split_instance.q_hat.device.type == 'cuda'
        
        # Test prediction on GPU
        x_test = torch.randn(3, 2).cuda()
        pred_sets = split_instance.predict(x_test)
        assert isinstance(pred_sets, list)

def test_memory_management(split_instance, calibration_data):
    # Test memory cleanup during calibration
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    split_instance.calibrate(calibration_data, alpha=0.1)
    post_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Memory should be cleaned up after calibration
    assert post_memory - initial_memory < 1024  # Small difference allowed

if __name__ == "__main__":
    pytest.main(["-v"])