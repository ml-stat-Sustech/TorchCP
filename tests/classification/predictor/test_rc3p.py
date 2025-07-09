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
    predictor.calibrate(cal_dataloader, None)

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

def test_evaluate_with_specific_instance(predictor, mock_dataset):
    """Test evaluate method with a specific test instance"""
    # First perform calibration
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha=0.1)
    
    # Take one sample from the existing dataset
    test_instance, test_label = mock_dataset[0]  # Get the first sample
    test_instance = test_instance.unsqueeze(0)  # Add batch dimension
    
    # Create a test dataset with this specific instance
    class TestDataset(Dataset):
        def __init__(self, x, labels):
            self.x = x
            self.labels = labels
            
        def __len__(self):
            return len(self.x)
            
        def __getitem__(self, idx):
            return self.x[idx], self.labels[idx]
    
    test_dataset = TestDataset(test_instance, torch.tensor([test_label]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # Evaluate with the specific test instance
    result_dict = predictor.evaluate(test_dataloader)
    
    # Verify the results
    assert isinstance(result_dict, dict)
    assert 'coverage_rate' in result_dict
    assert 'average_size' in result_dict
    assert 0 <= result_dict['coverage_rate'] <= 1
    assert 0 <= result_dict['average_size'] <= 3
    
    # Test with multiple instances
    multiple_instances = torch.randn(5, 3)  # 5 instances
    test_dataset_multiple = TestDataset(multiple_instances, torch.randint(0, 3, (5,)))
    test_dataloader_multiple = torch.utils.data.DataLoader(test_dataset_multiple, batch_size=2)
    
    result_dict_multiple = predictor.evaluate(test_dataloader_multiple)
    
    assert isinstance(result_dict_multiple, dict)
    assert 'coverage_rate' in result_dict_multiple
    assert 'average_size' in result_dict_multiple

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

    predictor.calculate_threshold(logits, labels, None)