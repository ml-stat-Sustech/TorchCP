# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.classification.loss.cd import CDLoss
from torchcp.classification.predictor import SplitPredictor as Predictor
from torchcp.classification.score import LAC


@pytest.fixture
def ds_instance():
    predictor = Predictor(LAC())
    epsilon = 1e-4
    return CDLoss(predictor, epsilon)


def test_init(ds_instance):
    ds = ds_instance
    assert isinstance(ds.predictor, Predictor)
    assert ds.epsilon == 1e-4


def test_invalid_epsilon():
    with pytest.raises(ValueError):
        CDLoss(Predictor(LAC()), 0)


def test_forward_with_different_epsilon():
    predictor = Predictor(LAC())
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))
    epsilons = [1e-3, 1e-2, 1e-1]
    for epsilon in epsilons:
        cd_loss = CDLoss(predictor, epsilon)
        loss = cd_loss.forward(logits, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])


def test_forward_with_edge_cases():
    predictor = Predictor(LAC())
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))

    # Test with very small epsilon
    ds_loss = CDLoss(predictor, 1e-10)
    loss = ds_loss.forward(logits, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])

    # Test with very large epsilon
    ds_loss = CDLoss(predictor, 1e+10)
    loss = ds_loss.forward(logits, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])
