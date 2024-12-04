import pytest
import torch
from torchcp.classification.score.knn import KNN

@pytest.fixture
def sample_data():
    return {
        'features': torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=torch.float32),
        'labels': torch.tensor([0, 1, 0, 1]),
        'num_classes': 2
    }

def test_initialization(sample_data):
    # Test default initialization
    knn = KNN(sample_data['features'], sample_data['labels'], sample_data['num_classes'])
    assert knn.k == 1
    assert knn.p == 2
    assert knn.batch is None
    
    # Test custom initialization
    knn = KNN(sample_data['features'], sample_data['labels'], 
              num_classes=2, k=3, p="cosine", batch=2)
    assert knn.k == 3
    assert knn.p == "cosine"
    assert knn.batch == 2

def test_invalid_initialization(sample_data):
    # Test invalid k
    with pytest.raises(ValueError, match="k must be an integer.*"):
        KNN(sample_data['features'], sample_data['labels'], 
            sample_data['num_classes'], k=0)
    with pytest.raises(ValueError, match="k must be an integer.*"):
        KNN(sample_data['features'], sample_data['labels'], 
            sample_data['num_classes'], k=1.5)
    
    # Test invalid p
    with pytest.raises(ValueError, match="p must be a positive float.*"):
        KNN(sample_data['features'], sample_data['labels'], 
            sample_data['num_classes'], p=-1)
    with pytest.raises(ValueError, match="p must be a positive float.*"):
        KNN(sample_data['features'], sample_data['labels'], 
            sample_data['num_classes'], p="invalid")
    
    # Test invalid batch
    with pytest.raises(ValueError, match="batch must be None.*"):
        KNN(sample_data['features'], sample_data['labels'], 
            sample_data['num_classes'], batch=0)

def test_euclidean_distance(sample_data):
    knn = KNN(sample_data['features'], sample_data['labels'], 
              sample_data['num_classes'], p=2)
    test_features = torch.tensor([[1.5, 2.5]], dtype=torch.float32)
    scores = knn(test_features)
    
    assert scores.shape == (1, sample_data['num_classes'])
    assert not torch.any(torch.isnan(scores))

def test_cosine_similarity(sample_data):
    knn = KNN(sample_data['features'], sample_data['labels'], 
              sample_data['num_classes'], p="cosine")
    test_features = torch.tensor([[1.5, 2.5]], dtype=torch.float32)
    scores = knn(test_features)
    
    assert scores.shape == (1, sample_data['num_classes'])
    assert not torch.any(torch.isnan(scores))

def test_batch_processing(sample_data):
    # Test with different batch sizes
    for batch_size in [1, 2, len(sample_data['features'])]:
        knn = KNN(sample_data['features'], sample_data['labels'], 
                  sample_data['num_classes'], batch=batch_size)
        test_features = torch.randn(5, 2)
        scores = knn(test_features)
        assert scores.shape == (5, sample_data['num_classes'])

def test_device_handling(sample_data):
    if torch.cuda.is_available():
        # Test GPU usage
        knn = KNN(sample_data['features'], sample_data['labels'], 
                  sample_data['num_classes'])
        assert knn.device.type == "cuda"
        assert knn.train_features.device.type == "cuda"
        
        # Test predictions on GPU
        test_features = torch.randn(3, 2).cuda()
        scores = knn(test_features)
        assert scores.device.type == "cuda"

def test_edge_cases():
    # Test single training example
    features = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([0])
    knn = KNN(features, labels, num_classes=2)
    test_features = torch.tensor([[1.5, 2.5]], dtype=torch.float32)
    scores = knn(test_features)
    assert scores.shape == (1, 2)
    
    # Test k equals number of training examples
    features = torch.randn(5, 2)
    labels = torch.randint(0, 2, (5,))
    knn = KNN(features, labels, num_classes=2, k=5)
    scores = knn(test_features)
    assert scores.shape == (1, 2)

def test_numerical_stability():
    # Test with very large values
    features = torch.tensor([[1e10, 2e10], [-1e10, -2e10]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    knn = KNN(features, labels, num_classes=2)
    test_features = torch.tensor([[1.5e10, 2.5e10]], dtype=torch.float32)
    scores = knn(test_features)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))
    
    # Test with very small values
    features = torch.tensor([[1e-10, 2e-10], [-1e-10, -2e-10]], dtype=torch.float32)
    scores = knn(torch.tensor([[1.5e-10, 2.5e-10]], dtype=torch.float32))
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))
    
def test_empty_same_label_case(sample_data):
    knn = KNN(**sample_data)
    
    # Create test case where no training samples have same label
    features = torch.randn(2)
    labels = torch.tensor([3])  # Label not in training set
    
    scores = knn(features, labels)
    assert not torch.any(torch.isnan(scores))
    
def test_all_same_label_case(sample_data):
    # Modify sample data to have all same labels
    modified_data = {
        'features': sample_data['features'],
        'labels': torch.zeros_like(sample_data['labels']),  # All labels are 0
        'num_classes': sample_data['num_classes']
    }
    
    knn = KNN(**modified_data)
    features = torch.randn(2, 2)
    labels = torch.zeros(2)  # Test with matching labels
    
    scores = knn(features, labels)
    assert not torch.any(torch.isnan(scores))
    assert torch.all(torch.isfinite(scores))

if __name__ == "__main__":
    pytest.main(["-v"])