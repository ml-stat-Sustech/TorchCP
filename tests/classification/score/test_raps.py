import pytest
import torch
from torchcp.classification.scores.raps import RAPS
from torchcp.classification.scores.aps import APS

@pytest.fixture
def sample_data():
    return {
        'probs': torch.tensor([[0.1, 0.4, 0.5], 
                             [0.3, 0.3, 0.4]], dtype=torch.float32),
        'labels': torch.tensor([2, 1])
    }

def test_initialization():
    # Test valid initialization
    raps = RAPS(penalty=0.1, kreg=1)
    assert raps._RAPS__penalty == 0.1
    assert raps._RAPS__kreg == 1
    
    # Test invalid penalty
    with pytest.raises(ValueError, match="penalty.*nonnegative"):
        RAPS(penalty=-1)
    
    # Test invalid kreg
    with pytest.raises(TypeError, match="kreg.*nonnegative integer"):
        RAPS(kreg=-1)
    with pytest.raises(TypeError, match="kreg.*nonnegative integer"):
        RAPS(kreg=1.5)

def test_calculate_all_label_randomized(sample_data):
    torch.manual_seed(42)
    raps = RAPS(penalty=0.1, kreg=1, randomized=True)
    
    # First call
    scores1 = raps._calculate_all_label(sample_data['probs'])
    assert scores1.shape == sample_data['probs'].shape
    
    # Second call should be different due to randomization
    scores2 = raps._calculate_all_label(sample_data['probs'])
    assert not torch.allclose(scores1, scores2)

def test_calculate_all_label_deterministic(sample_data):
    raps = RAPS(penalty=0.1, kreg=1, randomized=False)
    
    # Multiple calls should give same results
    scores1 = raps._calculate_all_label(sample_data['probs'])
    scores2 = raps._calculate_all_label(sample_data['probs'])
    assert torch.allclose(scores1, scores2)

def test_calculate_single_label_randomized(sample_data):
    torch.manual_seed(42)
    raps = RAPS(penalty=0.1, kreg=1, randomized=True)
    
    # First call
    scores1 = raps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert scores1.shape == (2,)
    
    # Second call should be different
    scores2 = raps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert not torch.allclose(scores1, scores2)

def test_calculate_single_label_deterministic(sample_data):
    raps = RAPS(randomized=False, penalty=0.1, kreg=1)
    scores1 = raps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    scores2 = raps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert torch.allclose(scores1, scores2)

def test_device_compatibility():
    if torch.cuda.is_available():
        raps = RAPS(penalty=0.1, kreg=1)
        probs = torch.tensor([[0.1, 0.4, 0.5]], device='cuda')
        labels = torch.tensor([1], device='cuda')
        
        # Test all_label
        scores = raps._calculate_all_label(probs)
        assert scores.device.type == 'cuda'
        
        # Test single_label
        scores = raps._calculate_single_label(probs, labels)
        assert scores.device.type == 'cuda'

def test_edge_cases():
    raps = RAPS(penalty=0.1, kreg=1)
    
    # Test uniform probabilities
    uniform_probs = torch.ones(2, 3) / 3
    scores = raps._calculate_all_label(uniform_probs)
    assert not torch.any(torch.isnan(scores))
    
    # Test zero penalty
    raps_no_penalty = RAPS(penalty=0, kreg=1)
    scores = raps_no_penalty._calculate_all_label(uniform_probs)
    assert not torch.any(torch.isnan(scores))
    
    # Test kreg = 0
    raps_no_reg = RAPS(penalty=0.1, kreg=0)
    scores = raps_no_reg._calculate_all_label(uniform_probs)
    assert not torch.any(torch.isnan(scores))

def test_inheritance():
    raps = RAPS()
    assert isinstance(raps, APS)
    
    # Test inherited method
    probs = torch.tensor([[0.1, 0.4, 0.5]])
    indices, ordered, cumsum = raps._sort_sum(probs)
    assert indices.shape == probs.shape

if __name__ == "__main__":
    pytest.main(["-v"])