import pytest
import torch

from torchcp.classification.score.aps import APS
from torchcp.classification.score.margin import Margin


@pytest.fixture
def sample_data():
    return {
        'probs': torch.tensor([[0.1, 0.4, 0.5],
                               [0.3, 0.3, 0.4]], dtype=torch.float32),
        'labels': torch.tensor([2, 1])
    }


def test_initialization():
    # Test default initialization
    margin = Margin()
    assert margin.score_type == "softmax"

    # Test custom initialization
    margin = Margin(score_type="identity")
    assert margin.score_type == "identity"

    # Test inheritance
    assert isinstance(margin, APS)


def test_calculate_single_label(sample_data):
    margin = Margin()
    scores = margin._calculate_single_label(
        sample_data['probs'].clone(),  # Clone to avoid modifying original
        sample_data['labels']
    )

    assert scores.shape == (2,)
    # For first sample: max(0.1, 0.4) - 0.5 = 0.4 - 0.5 = -0.1
    # For second sample: max(0.3, 0.4) - 0.3 = 0.4 - 0.3 = 0.1
    expected = torch.tensor([-0.1, 0.1])
    assert torch.allclose(scores, expected, rtol=1e-5)


def test_calculate_all_label(sample_data):
    margin = Margin()
    scores = margin._calculate_all_label(sample_data['probs'])

    assert scores.shape == sample_data['probs'].shape

    # Verify first row calculations
    # For class 0: 0.5 - 0.1 = 0.4
    # For class 1: 0.5 - 0.4 = 0.1
    # For class 2: 0.4 - 0.5 = -0.1
    expected_first_row = torch.tensor([0.4, 0.1, -0.1])
    assert torch.allclose(scores[0], expected_first_row, rtol=1e-5)


def test_edge_cases():
    margin = Margin()

    # Test uniform probabilities
    uniform_probs = torch.ones(2, 3) / 3
    scores_single = margin._calculate_single_label(
        uniform_probs.clone(),
        torch.tensor([0, 1])
    )
    assert torch.allclose(scores_single, torch.zeros_like(scores_single))

    scores_all = margin._calculate_all_label(uniform_probs)
    assert torch.allclose(scores_all, torch.zeros_like(scores_all))

    # Test one-hot probabilities
    one_hot = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
    scores_single = margin._calculate_single_label(
        one_hot.clone(),
        torch.tensor([0, 1])
    )
    assert torch.all(scores_single < 0)  # Maximum margin


def test_device_handling():
    if torch.cuda.is_available():
        margin = Margin()
        probs = torch.tensor([[0.1, 0.4, 0.5]], device='cuda')
        labels = torch.tensor([1], device='cuda')

        scores_single = margin._calculate_single_label(probs.clone(), labels)
        assert scores_single.device.type == 'cuda'

        scores_all = margin._calculate_all_label(probs)
        assert scores_all.device.type == 'cuda'


def test_documentation_example():
    margin = Margin(score_type="softmax")
    probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
    labels = torch.tensor([2, 1])

    scores_single = margin._calculate_single_label(probs.clone(), labels)
    assert scores_single.shape == (2,)

    scores_all = margin._calculate_all_label(probs)
    assert scores_all.shape == probs.shape


def test_numerical_stability():
    margin = Margin()

    # Test with very small probabilities
    small_probs = torch.tensor([[1e-10, 1e-9, 1 - 2e-9]])
    scores = margin._calculate_all_label(small_probs)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))

    # Test with very close probabilities
    close_probs = torch.tensor([[0.33333, 0.33334, 0.33333]])
    scores = margin._calculate_all_label(close_probs)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))


if __name__ == "__main__":
    pytest.main(["-v"])
