import pytest
import torch
from torchcp.classification.scores.aps import APS
from torchcp.classification.scores.thr import THR

@pytest.fixture
def sample_probs():
    return {
        'basic': torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]], dtype=torch.float32),
        'uniform': torch.ones(2, 3) / 3,
        'one_hot': torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
        'small': torch.tensor([[1e-6, 1-2e-6, 1e-6]], dtype=torch.float32)
    }

def test_initialization():
    # Test default initialization
    aps = APS()
    assert aps.score_type == "softmax"
    assert aps.randomized == True
    
    # Test custom initialization
    aps = APS(score_type="identity", randomized=False)
    assert aps.score_type == "identity"
    assert aps.randomized == False
    
    # Test inheritance
    assert isinstance(aps, THR)

def test_sort_sum(sample_probs):
    aps = APS()
    probs = sample_probs['basic']
    
    indices, ordered, cumsum = aps._sort_sum(probs)
    
    # Check shapes
    assert indices.shape == probs.shape
    assert ordered.shape == probs.shape
    assert cumsum.shape == probs.shape
    
    # Check sorting
    assert torch.all(torch.diff(ordered[0]) <= 0)  # Descending order
    
    # Check cumsum
    assert torch.allclose(cumsum[:, -1], torch.ones(2))

def test_calculate_all_label(sample_probs):
    aps = APS(randomized=False)  # Deterministic for testing
    
    for name, probs in sample_probs.items():
        scores = aps._calculate_all_label(probs)
        assert scores.shape == probs.shape
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

def test_randomization():
    probs = torch.tensor([[0.3, 0.3, 0.4]])
    
    # Test deterministic
    aps_det = APS(randomized=False)
    scores_det1 = aps_det._calculate_all_label(probs)
    scores_det2 = aps_det._calculate_all_label(probs)
    assert torch.allclose(scores_det1, scores_det2)
    
    # Test randomized with fixed seed
    aps_rand = APS(randomized=True)
    torch.manual_seed(42)
    scores_rand1 = aps_rand._calculate_all_label(probs)
    torch.manual_seed(42)
    scores_rand2 = aps_rand._calculate_all_label(probs)
    assert torch.allclose(scores_rand1, scores_rand2)

def test_error_handling():
    aps = APS()
    
    # Test 1D input
    with pytest.raises(ValueError, match="Input probabilities must be 2D"):
        aps._calculate_all_label(torch.tensor([0.1, 0.9]))
    
    # Test 3D input
    with pytest.raises(ValueError, match="Input probabilities must be 2D"):
        aps._calculate_all_label(torch.ones(2, 3, 4))
    

def test_device_compatibility():
    if torch.cuda.is_available():
        aps = APS()
        probs = torch.tensor([[0.1, 0.4, 0.5]], device='cuda')
        scores = aps._calculate_all_label(probs)
        assert scores.device == probs.device
        
        # Test randomization on GPU
        aps_rand = APS(randomized=True)
        rand_scores = aps_rand._calculate_all_label(probs)
        assert rand_scores.device == probs.device

def test_dtype_compatibility():
    aps = APS()
    dtypes = [torch.float32, torch.float64]
    
    for dtype in dtypes:
        probs = torch.tensor([[0.1, 0.4, 0.5]], dtype=dtype)
        scores = aps._calculate_all_label(probs)
        assert scores.dtype == dtype

def test_documentation_example():
    aps = APS(score_type="softmax", randomized=True)
    probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
    scores = aps._calculate_all_label(probs)
    assert scores.shape == probs.shape

if __name__ == "__main__":
    pytest.main(["-v"])