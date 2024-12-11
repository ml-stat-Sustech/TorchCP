# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.classification.score.aps import APS
from torchcp.classification.score.thr import THR


@pytest.fixture
def sample_data():
    return {
        'probs': torch.tensor([[0.1, 0.4, 0.5],
                               [0.3, 0.3, 0.4]], dtype=torch.float32),
        'labels': torch.tensor([2, 1])
    }


def test_initialization():
    # Test valid initialization
    aps = APS(score_type="softmax", randomized=True)
    assert aps.score_type == "softmax"
    assert aps.randomized == True

    # Test invalid score_type
    with pytest.raises(ValueError, match="Score type .* is not implemented"):
        APS(score_type="invalid")


def test_calculate_all_label_randomized(sample_data):
    torch.manual_seed(42)
    aps = APS(score_type="softmax", randomized=True)

    # First call
    scores1 = aps._calculate_all_label(sample_data['probs'])
    assert scores1.shape == sample_data['probs'].shape

    # Second call should be different due to randomization
    scores2 = aps._calculate_all_label(sample_data['probs'])
    assert not torch.allclose(scores1, scores2)


def test_calculate_all_label_deterministic(sample_data):
    aps = APS(score_type="softmax", randomized=False)

    # Multiple calls should give same results
    scores1 = aps._calculate_all_label(sample_data['probs'])
    scores2 = aps._calculate_all_label(sample_data['probs'])
    assert torch.allclose(scores1, scores2)


def test_sort_sum(sample_data):
    aps = APS(score_type="softmax", randomized=False)
    indices, ordered, cumsum = aps._sort_sum(sample_data['probs'])

    # Test shapes
    assert indices.shape == sample_data['probs'].shape
    assert ordered.shape == sample_data['probs'].shape
    assert cumsum.shape == sample_data['probs'].shape

    # Test sorting is correct
    sorted_probs = torch.gather(sample_data['probs'], 1, indices)
    assert torch.all(torch.diff(sorted_probs, dim=1) <= 0)  # Descending order

    # Test cumulative sum
    assert torch.allclose(cumsum, torch.cumsum(ordered, dim=-1))


def test_calculate_single_label_randomized(sample_data):
    torch.manual_seed(42)
    aps = APS(score_type="softmax", randomized=True)

    # First call
    scores1 = aps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert scores1.shape == (2,)

    # Second call should be different due to randomization
    scores2 = aps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert not torch.allclose(scores1, scores2)


def test_calculate_single_label_deterministic(sample_data):
    aps = APS(score_type="softmax", randomized=False)

    scores1 = aps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    scores2 = aps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert torch.allclose(scores1, scores2)


def test_device_compatibility():
    if torch.cuda.is_available():
        aps = APS(score_type="softmax")
        probs = torch.tensor([[0.1, 0.4, 0.5]], device='cuda')
        labels = torch.tensor([1], device='cuda')

        # Test _calculate_all_label
        scores = aps._calculate_all_label(probs)
        assert scores.device.type == 'cuda'

        # Test _calculate_single_label
        scores = aps._calculate_single_label(probs, labels)
        assert scores.device.type == 'cuda'


def test_edge_cases():
    aps = APS(score_type="softmax")

    # Test uniform probabilities
    uniform_probs = torch.ones(2, 3) / 3
    scores = aps._calculate_all_label(uniform_probs)
    assert not torch.any(torch.isnan(scores))

    # Test one-hot probabilities
    one_hot = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
    scores = aps._calculate_all_label(one_hot)
    assert not torch.any(torch.isnan(scores))

    # Test invalid input dimensions
    with pytest.raises(ValueError, match="Input probabilities must be 2D"):
        aps._calculate_all_label(torch.ones(3))
    with pytest.raises(ValueError, match="Input probabilities must be 2D"):
        aps._calculate_all_label(torch.ones(2, 3, 4))


def test_documentation_example():
    aps = APS(score_type="softmax", randomized=True)
    probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])

    scores_all = aps._calculate_all_label(probs)
    assert scores_all.shape == probs.shape

    labels = torch.tensor([2, 1])
    scores_single = aps._calculate_single_label(probs, labels)
    assert scores_single.shape == (2,)


if __name__ == "__main__":
    pytest.main(["-v"])
