import pytest
import torch
import numpy as np
from torchcp.classification.utils.metrics import (coverage_rate, average_size, CovGap, VioClasses,
                        DiffViolation, SSCV, WSC, Metrics)

@pytest.fixture
def setup_basic_data():
    """Create basic test data"""
    prediction_sets = torch.tensor([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=torch.bool)
    labels = torch.tensor([0, 0, 1, 2])
    return prediction_sets, labels

def test_coverage_rate_default(setup_basic_data):
    """Test default coverage rate calculation"""
    prediction_sets, labels = setup_basic_data
    cvg = coverage_rate(prediction_sets, labels)
    assert isinstance(cvg, float)
    assert 0 <= cvg <= 1
    # In this example, 3/4 samples should be correctly covered
    assert pytest.approx(cvg) == 1

def test_coverage_rate_macro(setup_basic_data):
    """Test macro-averaged coverage rate calculation"""
    prediction_sets, labels = setup_basic_data
    cvg = coverage_rate(prediction_sets, labels, coverage_type="macro", num_classes=4)
    assert isinstance(cvg, float)
    assert 0 <= cvg <= 1

def test_coverage_rate_errors(setup_basic_data):
    """Test error handling in coverage rate calculation"""
    prediction_sets, labels = setup_basic_data
    
    # Test dimension mismatch error
    with pytest.raises(ValueError):
        coverage_rate(prediction_sets[:-1], labels)
    
    # Test invalid coverage type
    with pytest.raises(ValueError):
        coverage_rate(prediction_sets, labels, coverage_type="invalid")
    
    # Test missing num_classes in macro mode
    with pytest.raises(ValueError):
        coverage_rate(prediction_sets, labels, coverage_type="macro")

def test_average_size(setup_basic_data):
    """Test average set size calculation"""
    prediction_sets, _ = setup_basic_data
    avg_size = average_size(prediction_sets)
    assert isinstance(avg_size, torch.Tensor)
    assert avg_size.item() > 0

def test_covgap(setup_basic_data):
    """Test class-conditional coverage gap calculation"""
    prediction_sets, labels = setup_basic_data
    alpha = 0.1
    num_classes = 4
    
    with pytest.raises(ValueError):
        CovGap(prediction_sets[:-1], labels, alpha, num_classes)
    
    # Test basic functionality
    gap = CovGap(prediction_sets, labels, alpha, num_classes)
    assert isinstance(gap, float)
    assert gap >= 0
    
    # Test with shot_idx
    shot_idx = [0, 1]
    gap_shot = CovGap(prediction_sets, labels, alpha, num_classes, shot_idx)
    assert isinstance(gap_shot, float)

def test_vioclasses(setup_basic_data):
    """Test calculation of number of violated classes"""
    prediction_sets, labels = setup_basic_data
    alpha = 0.1
    num_classes = 4
    
    with pytest.raises(ValueError):
        VioClasses(prediction_sets[:-1], labels, alpha, num_classes)
    
    
    vio = VioClasses(prediction_sets, labels, alpha, num_classes)
    assert isinstance(vio, int)
    assert vio >= 0
    assert vio <= num_classes

@pytest.fixture
def setup_diff_violation_data(setup_basic_data):
    """Create data for difficulty-stratified violation testing"""
    prediction_sets, labels = setup_basic_data
    logits = torch.randn(4, 4)  # Random logits
    return logits, prediction_sets, labels

def test_diff_violation(setup_diff_violation_data):
    """Test difficulty-stratified violation calculation"""
    logits, prediction_sets, labels = setup_diff_violation_data
    alpha = 0.1
    
    with pytest.raises(ValueError):
        DiffViolation(logits, prediction_sets[:-1], labels, alpha)
        
    with pytest.raises(TypeError):
        DiffViolation(logits, prediction_sets, labels, alpha, None)
    
    diff_vio, stats = DiffViolation(logits, prediction_sets, labels, alpha)
    assert isinstance(diff_vio, float)
    assert isinstance(stats, dict)
    assert diff_vio >= -1

def test_sscv(setup_basic_data):
    """Test size-stratified coverage violation calculation"""
    

        
    prediction_sets, labels = setup_basic_data
    alpha = 0.1
        
    
    with pytest.raises(ValueError, match="The number of prediction sets must be equal to the number of labels"):
        SSCV(prediction_sets[:-1], labels, alpha)
    with pytest.raises(ValueError, match="stratified_size must be a non-empty list"):
        SSCV(prediction_sets, labels, alpha, stratified_size=None)
    
    # Test basic functionality
    sscv = SSCV(prediction_sets, labels, alpha)
    assert isinstance(sscv, float)
    assert sscv >= -1
    
    # Test with custom stratified size
    custom_size = [[0, 2], [3, 4]]
    sscv_custom = SSCV(prediction_sets, labels, alpha, stratified_size=custom_size)
    assert isinstance(sscv_custom, float)

@pytest.fixture
def setup_large_data():
   """Create large test data for metrics testing"""
   n_samples = 1000
   n_classes = 100
   n_features = 512
   
   # Create prediction sets and labels
   prediction_sets = torch.randint(0, 2, (n_samples, n_classes), dtype=torch.bool)
   labels = torch.randint(0, n_classes, (n_samples,))
   features = torch.randn(n_samples, n_features)
   
   return features, prediction_sets, labels, 

def test_wsc_normal_case():
    """Test WSC function with normal inputs"""
    # Prepare test data
    n_samples, n_features, n_classes = 1000, 10, 3
    features = torch.randn(n_samples, n_features)
    prediction_sets = torch.ones(n_samples, n_classes, dtype=torch.bool)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Execute function
    result = WSC(features, prediction_sets, labels)
    
    # Verify results
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_wsc_input_shapes():
    """Test error handling for different input shapes"""
    # Prepare data with incorrect shapes
    features_1d = torch.randn(10)
    features_3d = torch.randn(10, 5, 2)
    prediction_sets_1d = torch.ones(10, dtype=torch.bool)
    labels_2d = torch.zeros(10, 2)
    
    # Base data with correct shapes
    features = torch.randn(10, 5)
    prediction_sets = torch.ones(10, 3, dtype=torch.bool)
    labels = torch.zeros(10)
    
    # Test invalid features shape
    with pytest.raises(ValueError, match="features must be 2D tensor"):
        WSC(features_1d, prediction_sets, labels)
    
    with pytest.raises(ValueError, match="features must be 2D tensor"):
        WSC(features_3d, prediction_sets, labels)
    
    # Test invalid prediction_sets shape
    with pytest.raises(ValueError, match="prediction_sets must be 2D tensor"):
        WSC(features, prediction_sets_1d, labels)
    
    # Test invalid labels shape
    with pytest.raises(ValueError, match="labels must be 1D tensor"):
        WSC(features, prediction_sets, labels_2d)

def test_wsc_sample_size_mismatch():
    """Test cases where sample sizes don't match"""
    features = torch.randn(10, 5)
    prediction_sets = torch.ones(8, 3, dtype=torch.bool)  # Mismatched sample size
    labels = torch.zeros(10)
    
    with pytest.raises(ValueError, match="Number of samples mismatch"):
        WSC(features, prediction_sets, labels)
    
    # Test mismatched labels sample size
    labels_mismatched = torch.zeros(8)
    with pytest.raises(ValueError, match="Number of samples mismatch"):
        WSC(features, prediction_sets, labels_mismatched)

def test_wsc_parameter_ranges():
    """Test parameter range validation"""
    # Prepare base correct data
    features = torch.randn(10, 5)
    prediction_sets = torch.ones(10, 3, dtype=torch.bool)
    labels = torch.zeros(10)
    
    # Test delta range
    with pytest.raises(ValueError, match="delta must be between 0 and 1"):
        WSC(features, prediction_sets, labels, delta=1.5)
    
    with pytest.raises(ValueError, match="delta must be between 0 and 1"):
        WSC(features, prediction_sets, labels, delta=0)
    
    # Test test_fraction range
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        WSC(features, prediction_sets, labels, test_fraction=1.2)
    
    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        WSC(features, prediction_sets, labels, test_fraction=0)
    
    # Test M value
    with pytest.raises(ValueError, match="M must be positive"):
        WSC(features, prediction_sets, labels, M=0)

def test_wsc_class_mismatch():
    """Test cases where number of classes don't match"""
    features = torch.randn(10, 5)
    prediction_sets = torch.ones(10, 5, dtype=torch.bool)  # 5 classes
    labels = torch.tensor([0, 1, 0, 1, 2, 0, 1, 0, 1, 0])  # Only 3 unique classes
    
    with pytest.raises(ValueError, match="Number of classes mismatch"):
        WSC(features, prediction_sets, labels)


def test_wsc_with_fixture(setup_large_data):
    """Test normal functionality using fixture"""
    features, prediction_sets, labels = setup_large_data
    result = WSC(features, prediction_sets, labels)
    assert isinstance(result, float)
    assert 0 <= result <= 1
    
def test_metrics_class():
    """Test Metrics class functionality"""
    metrics = Metrics()
    
    # Test valid metric
    assert callable(metrics('coverage_rate'))
    
    # Test invalid metric
    with pytest.raises(NameError):
        metrics('invalid_metric')

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    # Test with all-zero prediction sets
    zero_preds = torch.zeros((5, 3), dtype=torch.bool)
    labels = torch.tensor([0, 1, 2, 1, 0])
    
    cvg = coverage_rate(zero_preds, labels)
    assert cvg == 0.0
    
    # Test with all-one prediction sets
    one_preds = torch.ones((5, 3), dtype=torch.bool)
    cvg = coverage_rate(one_preds, labels)
    assert cvg == 1.0

def test_device_compatibility(setup_basic_data):
    """Test compatibility across different devices"""
    prediction_sets, labels = setup_basic_data
    if torch.cuda.is_available():
        prediction_sets = prediction_sets.cuda()
        labels = labels.cuda()
        cvg = coverage_rate(prediction_sets, labels)
        assert isinstance(cvg, float)

@pytest.fixture
def create_random_data():
    """Helper function to create random test data"""
    def _create_random_data(n_samples=100, n_classes=10):
        prediction_sets = torch.randint(0, 2, (n_samples, n_classes)).bool()
        labels = torch.randint(0, n_classes, (n_samples,))
        return prediction_sets, labels
    return _create_random_data

def test_random_data(create_random_data):
    """Test with randomly generated data"""
    prediction_sets, labels = create_random_data()
    cvg = coverage_rate(prediction_sets, labels)
    assert isinstance(cvg, float)
    assert 0 <= cvg <= 1