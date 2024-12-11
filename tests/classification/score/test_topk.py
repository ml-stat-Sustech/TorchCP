# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.classification.score.thr import THR
from torchcp.classification.score.topk import TOPK


@pytest.fixture
def sample_data():
    return {
        'probs': torch.tensor([[0.1, 0.4, 0.5],
                               [0.3, 0.3, 0.4]], dtype=torch.float32),
        'labels': torch.tensor([2, 1])
    }


def test_initialization():
    # Test default initialization
    topk = TOPK()
    assert topk.randomized == True
    assert topk.score_type == "softmax"

    # Test custom initialization
    topk = TOPK(randomized=False, score_type="identity")
    assert topk.randomized == False
    assert topk.score_type == "identity"


def test_inheritance():
    topk = TOPK()
    assert isinstance(topk, THR)


def test_calculate_all_label_randomized(sample_data):
    torch.manual_seed(42)
    topk = TOPK(randomized=True)

    # First call
    scores1 = topk._calculate_all_label(sample_data['probs'])
    assert scores1.shape == sample_data['probs'].shape

    # Second call should be different due to randomization
    scores2 = topk._calculate_all_label(sample_data['probs'])
    assert not torch.allclose(scores1, scores2)


def test_calculate_all_label_deterministic(sample_data):
    topk = TOPK(randomized=False)

    # Multiple calls should give same results
    scores1 = topk._calculate_all_label(sample_data['probs'])
    scores2 = topk._calculate_all_label(sample_data['probs'])
    assert torch.allclose(scores1, scores2)


def test_sort_sum(sample_data):
    topk = TOPK()
    indices, ones, cumsum = topk._sort_sum(sample_data['probs'])

    # Test shapes
    assert indices.shape == sample_data['probs'].shape
    assert ones.shape == sample_data['probs'].shape
    assert cumsum.shape == sample_data['probs'].shape

    # Test sorting is correct
    sorted_probs = torch.gather(sample_data['probs'], 1, indices)
    assert torch.all(torch.diff(sorted_probs, dim=1) <= 0)  # Descending order

    # Test ones and cumsum
    assert torch.all(ones == 1)
    assert torch.all(cumsum == torch.arange(1, cumsum.shape[1] + 1))


def test_device_compatibility():
    if torch.cuda.is_available():
        topk = TOPK()
        probs = torch.tensor([[0.1, 0.4, 0.5]], device='cuda')

        # Test _calculate_all_label
        scores = topk._calculate_all_label(probs)
        assert scores.device.type == 'cuda'

        # Test _sort_sum
        indices, ones, cumsum = topk._sort_sum(probs)
        assert indices.device.type == 'cuda'
        assert ones.device.type == 'cuda'
        assert cumsum.device.type == 'cuda'


def test_edge_cases():
    topk = TOPK()

    # Test uniform probabilities
    uniform_probs = torch.ones(2, 3) / 3
    scores = topk._calculate_all_label(uniform_probs)
    assert not torch.any(torch.isnan(scores))

    # Test one-hot probabilities
    one_hot = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
    scores = topk._calculate_all_label(one_hot)
    assert not torch.any(torch.isnan(scores))


def test_documentation_example():
    topk = TOPK(score_type="softmax", randomized=True)
    probs = torch.tensor([[0.1, 0.4, 0.5], [0.3, 0.3, 0.4]])
    scores = topk._calculate_all_label(probs)
    assert scores.shape == probs.shape


def test_numerical_stability():
    topk = TOPK()

    # Test with very small probabilities
    small_probs = torch.tensor([[1e-10, 1e-9, 1 - 2e-9]])
    scores = topk._calculate_all_label(small_probs)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))

    # Test with very similar probabilities
    close_probs = torch.tensor([[0.33333, 0.33334, 0.33333]])
    scores = topk._calculate_all_label(close_probs)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))


def test_calculate_single_label():
    # Fixture data
    probs = torch.tensor([[0.1, 0.4, 0.5],
                          [0.3, 0.3, 0.4]], dtype=torch.float32)
    labels = torch.tensor([2, 1])

    # Test randomized=True
    torch.manual_seed(42)
    topk = TOPK(randomized=True)
    scores1 = topk._calculate_single_label(probs, labels)
    assert scores1.shape == (2,)

    # Test randomization effect
    scores2 = topk._calculate_single_label(probs, labels)
    assert not torch.allclose(scores1, scores2)

    # Test deterministic mode
    topk_det = TOPK(randomized=False)
    scores3 = topk_det._calculate_single_label(probs, labels)
    scores4 = topk_det._calculate_single_label(probs, labels)
    assert torch.allclose(scores3, scores4)

    # Test device handling
    if torch.cuda.is_available():
        probs_cuda = probs.cuda()
        labels_cuda = labels.cuda()
        scores_cuda = topk._calculate_single_label(probs_cuda, labels_cuda)
        assert scores_cuda.device.type == 'cuda'

    # Test label indexing
    single_prob = torch.tensor([[0.1, 0.4, 0.5]])
    single_label = torch.tensor([0])  # Test first position
    scores = topk._calculate_single_label(single_prob, single_label)
    assert scores.shape == (1,)


if __name__ == "__main__":
    pytest.main(["-v", "--cov=topk", "--cov-report=term-missing"])
