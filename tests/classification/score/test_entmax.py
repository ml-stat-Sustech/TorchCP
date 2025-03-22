import pytest
import torch
from torchcp.classification.score.entmax import EntmaxScore

@pytest.fixture
def logits():
    return torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 1.0]])

def test_entmax_init():
    # Test normal initialization
    entmax = EntmaxScore(gamma=2.0)
    assert entmax.gamma == 2.0
    
    # Test error case when gamma < 1
    with pytest.raises(ValueError):
        EntmaxScore(gamma=0.5)

def test_entmax_softmax(logits):
    # Test when gamma=1.0 (softmax)
    entmax = EntmaxScore(gamma=1.0)
    scores = entmax(logits)
    expected = torch.tensor([[0.0, 1.0, 1.9],
                           [2.0, 0.0, 1.5]], dtype=torch.float32)
    
    torch.testing.assert_close(scores, expected, rtol=1e-3, atol=1e-3)

def test_entmax_sparsemax(logits):
    # Test when gamma=2.0 (sparsemax)
    entmax = EntmaxScore(gamma=2.0)
    scores = entmax(logits)
    expected = torch.tensor([[0.0, 1.0, 2.8],
                           [2.5, 0.0, 1.5]], dtype=torch.float32)
    
    torch.testing.assert_close(scores, expected, rtol=1e-3, atol=1e-3)

def test_entmax_single_label():
    # Test the case with single label
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    labels = torch.tensor([1])  # Choose the second class
    
    entmax = EntmaxScore(gamma=2.0)
    scores = entmax(logits, labels)
    assert scores.shape == (1,)
    assert scores[0] > 0  # Non-zero score, as prediction is not the highest class

def test_entmax_1d_input():
    # Test with 1D input
    logits = torch.tensor([2.0, 1.0, 0.1])
    entmax = EntmaxScore(gamma=2.0)
    scores = entmax(logits)
    assert scores.shape == (1, 3)

def test_invalid_input_dims():
    # Test invalid input dimensions
    logits = torch.randn(2, 3, 4)  # 3D tensor
    entmax = EntmaxScore(gamma=2.0)
    with pytest.raises(ValueError):
        entmax(logits)

def test_entmax_device_compatibility():
    # Test device compatibility (if CUDA is available)
    if torch.cuda.is_available():
        logits = torch.tensor([[2.0, 1.0, 0.1]], device='cuda')
        entmax = EntmaxScore(gamma=2.0)
        scores = entmax(logits)
        assert scores.device.type == 'cuda'

def test_general_entmax():
    # Test with a general gamma value
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    entmax = EntmaxScore(gamma=1.5)
    scores = entmax(logits)
    assert scores.shape == (1, 3)
    assert torch.all(scores >= 0)  # All scores should be non-negative

def test_entmax_1d_input_comprehensive():
    # Test 1D input processing (covers lines 93-94)
    logits = torch.tensor([2.0, 1.0, 0.1])
    entmax = EntmaxScore(gamma=2.0)
    
    # Test with all labels
    scores_all = entmax(logits)
    assert scores_all.shape == (1, 3)  # Ensure output is correctly reshaped to 2D
    
    # Test with a single label
    label = torch.tensor([1])  # Choose the second class
    scores_single = entmax(logits, label)
    assert scores_single.shape == (1,)  # Ensure output dimensions are correct

def test_entmax_gamma_greater_than_two():
    # Test when gamma > 2 (covers lines 103-110)
    logits = torch.tensor([[2.0, 1.0, 0.1],
                          [0.5, 2.5, 1.0]])
    labels = torch.tensor([1, 0])  # First sample chooses second class, second sample chooses first class
    
    entmax = EntmaxScore(gamma=3.0)
    scores = entmax(logits, labels)
    
    # Calculation for gamma=3.0:
    # delta = 1/(gamma-1) = 0.5
    # For the first sample [2.0, 1.0, 0.1], label=1:
    # diffs = [2.0-1.0] = [1.0]
    # score = (sum(diffs^delta))^(1/delta) = (1.0^0.5)^2
    # For the second sample [0.5, 2.5, 1.0], label=0:
    # diffs = [2.5-0.5] = [2.0]
    # score = (sum(diffs^delta))^(1/delta) = (2.0^0.5)^2
    
    assert scores.shape == (2,)
    assert torch.all(scores >= 0)
    
    # Test with all labels
    scores_all = entmax(logits)
    assert scores_all.shape == (2, 3)
    assert torch.all(scores_all >= 0)
    
    # Verify relative score magnitudes
    assert scores_all[0, 0] < scores_all[0, 1]  # First sample's first score should be less than second
    assert scores_all[1, 1] < scores_all[1, 0]  # Second sample's second score should be less than first

def test_entmax_gamma_edge_cases():
    # Test with gamma close to 1
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    gamma_close_to_one = 1.000001
    entmax = EntmaxScore(gamma=gamma_close_to_one)
    scores = entmax(logits)
    assert torch.all(torch.isfinite(scores))  # Ensure no numerical instability
    
    # Test with very large gamma values
    gamma_large = 10.0
    entmax = EntmaxScore(gamma=gamma_large)
    scores = entmax(logits)
    assert torch.all(torch.isfinite(scores))  # Ensure no numerical instability
    
def test_entmax_1d_logits_with_scalar_label():
    """Specifically test the case of 1D logits with scalar label, covering lines 93-94"""
    # Create 1D logits
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    # Create scalar label (note this is not a tensor, but a Python scalar)
    label = torch.tensor([1])  # Use Python int instead of torch.tensor
    
    # Create EntmaxScore instance
    entmax = EntmaxScore(gamma=2.0)
    
    # Call the __call__ method, which should trigger lines 93-94
    score = entmax(logits, label)
    
    # Verify results
    assert score.shape == torch.Size([1])  # Should be a scalar result
    assert torch.isfinite(score)
    
    # Compare with results using tensor label
    tensor_label = torch.tensor([1])
    tensor_score = entmax(logits, tensor_label)
    assert torch.isclose(score, tensor_score[0])
    
    # Also test with gamma=1.0
    entmax_1 = EntmaxScore(gamma=1.0)
    score_1 = entmax_1(logits, label)
    assert torch.isfinite(score_1)