# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcp.classification.predictor.utils import build_DomainDetecor, MidFNN, FNN, Linear, SmallFNN, BigFNN, IW


@pytest.fixture
def mock_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def models_are_equal(model1, model2):
    if len(list(model1.children())) != len(list(model2.children())):
        return False
    for (layer1, layer2) in zip(model1.children(), model2.children()):
        if type(layer1) != type(layer2):
            return False
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(param1, param2):
            return False
    return True


def test_build_DomainDetecor(mock_device):
    torch.manual_seed(42)
    domain_detector = build_DomainDetecor(128, 2, mock_device)
    model_device = next(domain_detector.parameters()).device
    assert type(model_device) is type(mock_device) and model_device.index == mock_device.index
    torch.manual_seed(42)
    mid_fnn = MidFNN(128, 2).to(mock_device)
    assert models_are_equal(domain_detector, mid_fnn)


def test_fnn_initialization():
    fnn = FNN(128, 2, 64, 2)

    assert len(fnn.model) == 7
    assert type(fnn.model[0]) is nn.Linear and fnn.model[0].in_features == 128 and fnn.model[0].out_features == 64
    assert type(fnn.model[1]) is nn.ReLU
    assert type(fnn.model[2]) is nn.Dropout and fnn.model[2].p == 0.5
    assert type(fnn.model[3]) is nn.Linear and fnn.model[3].in_features == 64 and fnn.model[3].out_features == 64
    assert type(fnn.model[4]) is nn.ReLU
    assert type(fnn.model[5]) is nn.Dropout and fnn.model[5].p == 0.5
    assert type(fnn.model[6]) is nn.Linear and fnn.model[6].in_features == 64 and fnn.model[6].out_features == 2


def test_fnn_foward():
    fnn = FNN(128, 2, 64, 2)

    x = torch.randn(10, 128)
    out_train = fnn(x, True)
    out_eval = fnn(x, False)
    assert not torch.allclose(out_train, out_eval)

    fnn.model.eval()
    excepted_out = F.softmax(fnn.model(x), dim=1)
    assert torch.allclose(out_eval, excepted_out)

    fnn = FNN(128, 1, 64, 2)
    out_eval = fnn(x, False)
    fnn.model.eval()
    excepted_out = torch.sigmoid(fnn.model(x))
    assert torch.allclose(out_eval, excepted_out)


def test_fnn_children():
    linear = Linear(128, 3)
    assert len(linear.model) == 1
    assert linear.model[0].in_features == 128 and linear.model[0].out_features == 3

    linear = Linear(128, 3, 64)
    assert len(linear.model) == 1
    assert linear.model[0].in_features == 64 and linear.model[0].out_features == 3

    small_fnn = SmallFNN(128, 3, 64)
    assert len(small_fnn.model) == 4
    assert small_fnn.model[0].in_features == 128 and small_fnn.model[0].out_features == 64
    assert small_fnn.model[-1].out_features == 3

    big_fnn = BigFNN(256, 3, 128)
    assert len(big_fnn.model) == 13
    assert big_fnn.model[0].in_features == 256 and big_fnn.model[0].out_features == 128
    assert big_fnn.model[-1].out_features == 3


def test_iw(mock_device):
    torch.manual_seed(42)
    domain_detector = build_DomainDetecor(128, 2, mock_device)
    iw = IW(domain_detector)
    torch.manual_seed(42)
    mid_fnn = MidFNN(128, 2).to(mock_device)
    assert models_are_equal(iw.domain_detector, mid_fnn)

    x = torch.randn(10, 128).to(mock_device)
    probs = iw(x)
    excepted_out = iw.domain_detector(x)
    excepted_probs = excepted_out[:, 1] / excepted_out[:, 0]
    assert torch.allclose(probs, excepted_probs)

    torch.manual_seed(42)
    domain_detector = build_DomainDetecor(128, 1, mock_device)
    iw = IW(domain_detector)
    torch.manual_seed(42)
    mid_fnn = MidFNN(128, 1).to(mock_device)
    probs = iw(x)
    excepted_out = iw.domain_detector(x)
    excepted_probs = excepted_out / (1 - excepted_out)
    assert torch.allclose(probs, excepted_probs)
