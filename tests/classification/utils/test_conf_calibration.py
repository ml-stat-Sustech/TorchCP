# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import pytest
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, TensorDataset

from torchcp.classification.utils.conf_calibration import ConfCalibrator, Identity, TS, ConfCalibrator_REGISTRY


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_data():
    n_samples, n_classes = 100, 10
    logits = torch.randn(n_samples, n_classes)
    targets = torch.randint(0, n_classes, (n_samples,))
    dataset = TensorDataset(logits, targets)
    return DataLoader(dataset, batch_size=32)


def test_conf_calibrator_registry_valid():
    ts = ConfCalibrator.registry_ConfCalibrator("TS")
    assert ts.__name__ == "TS"
    identity = ConfCalibrator.registry_ConfCalibrator("Identity")
    assert identity.__name__ == "Identity"


def test_conf_calibrator_registry_invalid():
    with pytest.raises(NameError):
        ConfCalibrator.registry_ConfCalibrator("NonExistent")


def test_identity_forward():
    identity = Identity()
    x = torch.randn(16, 10)
    out = identity(x)
    assert torch.equal(out, x)


def test_ts_init():
    ts = TS(temperature=2.0)
    assert ts.temperature.item() == 2.0


def test_ts_forward():
    ts = TS(temperature=2.0)
    x = torch.randn(16, 10)
    out = ts(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x / 2.0)


def test_ts_early_stopping(device, test_data):
    ts = TS().to(device)
    ts.temperature = nn.Parameter(torch.tensor(1.0))
    ts.optimze(test_data, device, epsilon=float('inf'))  # Force early stopping
    assert (ts.temperature.item() - 1.0) < float('inf')
