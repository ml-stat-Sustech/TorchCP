import math
import pytest
import torch
from torch.utils.data import Dataset
from torchcp.classification.predictor import RC3PPredictor
from torchcp.classification.score import LAC

@pytest.fixture
def mock_dataset():
    class MyDataset(Dataset):
        def __init__(self):
            self.x = torch.randn(100, 3)
            self.labels = torch.randint(0, 3, (100,))

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.labels[idx]

    return MyDataset()

@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x

    return MockModel()

@pytest.fixture
def mock_score_function():
    return LAC(score_type="softmax")

@pytest.fixture
def predictor(mock_score_function, mock_model):
    return RC3PPredictor(mock_score_function, mock_model)

def test_valid_initialization(predictor, mock_score_function, mock_model):
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model
    assert not predictor._model.training
    assert predictor._device == next(mock_model.parameters()).device
    assert predictor.num_classes is None
    assert predictor.class_thresholds is None
    assert predictor.class_rank_limits is None

@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calibrate(predictor, mock_dataset, mock_score_function, mock_model, alpha):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha)
    
    assert predictor.num_classes == 3
    assert predictor.class_thresholds is not None
    assert predictor.class_rank_limits is not None
    assert predictor.class_thresholds.shape == torch.Size([3])
    assert predictor.class_rank_limits.shape == torch.Size([3])

@pytest.mark.parametrize("alpha", [0, 1, -0.1, 2])
def test_invalid_calibrate_alpha(predictor, mock_dataset, alpha):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    with pytest.raises(ValueError, match="alpha should be a value in"):
        predictor.calibrate(cal_dataloader, alpha)

def test_invalid_calibrate_model(mock_score_function, mock_dataset):
    predictor = RC3PPredictor(mock_score_function, None)
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    with pytest.raises(ValueError, match="Model is not defined"):
        predictor.calibrate(cal_dataloader, 0.1)

def test_calculate_threshold(predictor, mock_score_function):
    logits = torch.randn(100, 3)
    labels = torch.randint(0, 3, (100,))
    alpha = 0.1
    
    predictor.num_classes = 3
    predictor.calculate_threshold(logits, labels, alpha)
    
    assert predictor.class_thresholds is not None
    assert predictor.class_rank_limits is not None
    assert predictor.class_thresholds.shape == torch.Size([3])
    assert predictor.class_rank_limits.shape == torch.Size([3])
    assert torch.all(predictor.class_rank_limits <= 3)

def test_predict(predictor, mock_dataset):
    # First perform calibration
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha=0.1)
    
    # Test prediction
    pred_sets = predictor.predict(mock_dataset.x)
    
    assert pred_sets.shape == (100, 3)  # batch_size x num_classes
    assert pred_sets.dtype == torch.bool
    
def test_predict_with_logits(predictor, mock_dataset):
    # 首先进行校准
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha=0.1)
    
    # 生成测试logits
    logits = torch.randn(50, 3)
    pred_sets = predictor.predict_with_logits(logits)
    
    assert pred_sets.shape == (50, 3)
    assert pred_sets.dtype == torch.bool

def test_predict_without_calibration(predictor, mock_dataset):
    with pytest.raises(ValueError, match="Calibration not performed"):
        predictor.predict_with_logits(torch.randn(10, 3))

def test_predict_invalid_model(mock_score_function, mock_dataset):
    predictor = RC3PPredictor(mock_score_function, None)
    with pytest.raises(ValueError, match="Model is not defined"):
        predictor.predict(mock_dataset.x)

def test_evaluate(predictor, mock_dataset):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha=0.1)
    
    results = predictor.evaluate(cal_dataloader)
    
    assert 'coverage_rate' in results
    assert 'average_size' in results
    assert 0 <= results['coverage_rate'] <= 1
    assert 0 <= results['average_size'] <= 3  # num_classes
    
def test_calculate_threshold_empty_class(predictor, mock_score_function):
    """Test the case when a class has no samples"""
    # Create a special dataset where class 2 has no samples
    logits = torch.randn(100, 3)
    # Only use labels 0 and 1, completely excluding class 2
    labels = torch.randint(0, 2, (100,))  
    alpha = 0.1
    
    predictor.num_classes = 3
    predictor.calculate_threshold(logits, labels, alpha)
    
    assert predictor.class_thresholds is not None
    assert predictor.class_rank_limits is not None
    assert predictor.class_thresholds.shape == torch.Size([3])
    assert predictor.class_rank_limits.shape == torch.Size([3])
    # Check thresholds for class 2 (empty class)
    assert torch.isinf(predictor.class_thresholds[2])  # Should be infinity
    assert predictor.class_rank_limits[2] == 3  # Should be number of classes