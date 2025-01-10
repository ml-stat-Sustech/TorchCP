# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.classification.score.thr import THR


@pytest.fixture
def logits_2d():
    return torch.tensor([[2.0, 1.0, 0.1],
                         [0.5, 2.5, 1.0]])


@pytest.fixture
def logits_1d():
    return torch.tensor([2.0, 1.0, 0.1])


def test_thr_initialization():
    # Test default initialization
    thr = THR()
    assert thr.score_type == "softmax"

    # Test valid score types
    valid_types = ["softmax", "identity", "log_softmax", "log"]
    for score_type in valid_types:
        thr = THR(score_type=score_type)
        assert thr.score_type == score_type

    # Test invalid score type
    with pytest.raises(ValueError, match="Score type .* is not implemented"):
        THR(score_type="invalid")


def test_transform_functions():
    logits = torch.tensor([[2.0, 1.0], [0.5, 2.5]])

    # Test identity transform
    thr_identity = THR(score_type="identity")
    assert torch.allclose(thr_identity.transform(logits), logits)

    # Test softmax transform
    thr_softmax = THR(score_type="softmax")
    softmax_output = thr_softmax.transform(logits)
    assert torch.allclose(torch.sum(softmax_output, dim=1), torch.ones(2))

    # Test log_softmax transform
    thr_log_softmax = THR(score_type="log_softmax")
    log_softmax_output = thr_log_softmax.transform(logits)
    assert torch.all(log_softmax_output <= 0)


def test_call_method(logits_2d):
    thr = THR()

    # Test without labels
    scores = thr(logits_2d)
    assert scores.shape == logits_2d.shape

    # Test with labels
    labels = torch.tensor([0, 1])
    scores = thr(logits_2d, labels)
    assert scores.shape[0] == logits_2d.shape[0]


def test_dimension_handling(logits_1d, logits_2d):
    thr = THR()

    # Test 1D input
    scores_1d = thr(logits_1d)
    assert len(scores_1d.shape) == 2
    assert scores_1d.shape[0] == 1

    # Test 2D input
    scores_2d = thr(logits_2d)
    assert len(scores_2d.shape) == 2
    assert scores_2d.shape == logits_2d.shape

    # Test invalid dimensions
    invalid_logits = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="dimension of logits are at most 2"):
        thr(invalid_logits)


def test_numerical_stability():
    thr = THR()

    # Test very large values
    large_logits = torch.tensor([[1e10, 1e-10], [-1e10, 1e10]])
    scores = thr(large_logits)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))

    # Test very small values
    small_logits = torch.tensor([[1e-10, 1e-10], [1e-10, 1e-10]])
    scores = thr(small_logits)
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))


def test_device_compatibility():
    if torch.cuda.is_available():
        thr = THR()
        logits = torch.randn(2, 3).cuda()
        scores = thr(logits)
        assert scores.device == logits.device


def test_custom_transform_function(logits_2d):
    # Test simple custom function
    custom_func = lambda x: x * 2
    thr = THR(score_type=custom_func)
    scores = thr(logits_2d)
    # breakpoint()
    assert torch.allclose(scores, 1 - logits_2d * 2)

    # Test more complex custom function
    def complex_transform(x):
        return torch.sigmoid(x) + torch.relu(x)

    thr = THR(score_type=complex_transform)
    scores = thr(logits_2d)
    expected = 1 - (torch.sigmoid(logits_2d) + torch.relu(logits_2d))

    assert torch.allclose(scores, expected)

    # Test custom function preserves shape
    assert scores.shape == logits_2d.shape
