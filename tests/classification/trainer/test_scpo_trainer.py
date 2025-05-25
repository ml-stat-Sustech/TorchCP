# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import pytest

from torchcp.classification.loss.scpo import SCPOLoss
from torchcp.classification.trainer.scpo_trainer import SCPOTrainer
from torchcp.classification.trainer.model_zoo import SurrogateCPModel


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)

        def forward(self, x):
            return x

    return MockModel()


def test_scpo_init_valid_params(mock_model):
    scpo_trainer = SCPOTrainer(alpha=0.1, model=mock_model)

    assert type(scpo_trainer.model) is SurrogateCPModel
    assert scpo_trainer.model.base_model is mock_model
    assert type(scpo_trainer.optimizer) is torch.optim.Adam
    assert scpo_trainer.optimizer.param_groups[0]['lr'] == 0.1
    assert type(scpo_trainer.loss_fn) is SCPOLoss