# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.classification.score.aps import APS
from torchcp.classification.score.saps import SAPS


@pytest.fixture
def sample_data():
    return {
        'probs': torch.tensor([[0.1, 0.4, 0.5],
                               [0.3, 0.3, 0.4]], dtype=torch.float32),
        'labels': torch.tensor([2, 1])
    }


def test_initialization():
    # Test valid initialization
    saps = SAPS(weight=0.5, randomized=True)
    assert saps._SAPS__weight == 0.5
    assert saps.randomized == True
    assert saps.score_type == "softmax"

    # Test invalid weight
    with pytest.raises(ValueError, match="weight.*positive"):
        SAPS(weight=0)
    with pytest.raises(ValueError, match="weight.*positive"):
        SAPS(weight=-1)

    # Test invalid randomized type
    with pytest.raises(ValueError, match="randomized.*boolean"):
        SAPS(weight=0.5, randomized="True")


def test_calculate_all_label_randomized(sample_data):
    torch.manual_seed(42)
    saps = SAPS(weight=0.5, randomized=True)

    # First call
    scores1 = saps._calculate_all_label(sample_data['probs'])
    assert scores1.shape == sample_data['probs'].shape

    # Second call should be different due to randomization
    scores2 = saps._calculate_all_label(sample_data['probs'])
    assert not torch.allclose(scores1, scores2)


def test_calculate_all_label_deterministic(sample_data):
    saps = SAPS(weight=0.5, randomized=False)

    # Multiple calls should give same results
    scores1 = saps._calculate_all_label(sample_data['probs'])
    scores2 = saps._calculate_all_label(sample_data['probs'])
    assert torch.allclose(scores1, scores2)


def test_calculate_single_label_randomized(sample_data):
    torch.manual_seed(42)
    saps = SAPS(weight=0.5, randomized=True)

    # First call
    scores1 = saps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert scores1.shape == (2,)

    # Second call should be different
    scores2 = saps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert not torch.allclose(scores1, scores2)


def test_calculate_single_label_deterministic(sample_data):
    saps = SAPS(weight=0.5, randomized=False)

    scores1 = saps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    scores2 = saps._calculate_single_label(sample_data['probs'], sample_data['labels'])
    assert torch.allclose(scores1, scores2)


def test_first_rank_scores(sample_data):
    saps = SAPS(weight=0.5, randomized=False)

    # Test with label in first position
    first_pos_label = torch.tensor([0])  # First class
    first_pos_scores = saps._calculate_single_label(sample_data['probs'][:1], first_pos_label)
    assert first_pos_scores.shape == (1,)


def test_device_compatibility():
    if torch.cuda.is_available():
        saps = SAPS(weight=0.5)
        probs = torch.tensor([[0.1, 0.4, 0.5]], device='cuda')
        labels = torch.tensor([1], device='cuda')

        # Test all_label
        scores = saps._calculate_all_label(probs)
        assert scores.device.type == 'cuda'

        # Test single_label
        scores = saps._calculate_single_label(probs, labels)
        assert scores.device.type == 'cuda'


def test_edge_cases():
    saps = SAPS(weight=0.5)

    # Test uniform probabilities
    uniform_probs = torch.ones(2, 3) / 3
    scores = saps._calculate_all_label(uniform_probs)
    assert not torch.any(torch.isnan(scores))

    # Test one-hot probabilities
    one_hot = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
    scores = saps._calculate_all_label(one_hot)
    assert not torch.any(torch.isnan(scores))


def test_inheritance():
    saps = SAPS(weight=0.5)
    assert isinstance(saps, APS)

    # Test inherited method
    probs = torch.tensor([[0.1, 0.4, 0.5]])
    indices, ordered, cumsum = saps._sort_sum(probs)
    assert indices.shape == probs.shape


def test_documentation_example():
    saps = SAPS(weight=0.5, score_type="softmax", randomized=True)
    probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])

    scores_all = saps._calculate_all_label(probs)
    assert scores_all.shape == probs.shape

    labels = torch.tensor([2, 1])
    scores_single = saps._calculate_single_label(probs, labels)
    assert scores_single.shape == (2,)


if __name__ == "__main__":
    pytest.main(["-v"])
