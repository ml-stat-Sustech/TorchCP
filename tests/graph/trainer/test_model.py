# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torch_geometric.nn import GCNConv

from torchcp.graph.trainer.model import CFGNNModel, GNN_Multi_Layer


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, edge_index=None):
            return x

    return MockModel()

@pytest.fixture
def mock_cfgnn_model(mock_model):
    return CFGNNModel(mock_model, 10)


def test_CFGNNModel_is_base_model_frozen(mock_cfgnn_model):
    assert mock_cfgnn_model.is_base_model_frozen() is True

    for param in mock_cfgnn_model.base_model.parameters():
        param.requires_grad = True

    assert mock_cfgnn_model.is_base_model_frozen() is False


def test_GNN_Multi_Layer_initialization():
    
    # num_layers == 1
    model = GNN_Multi_Layer(10, 64, 10, num_layers=1)
    assert len(model.convs) == 1
    assert type(model.convs[0]) is GCNConv
    assert model.convs[0].in_channels == 10 and model.convs[0].out_channels == 10

    # num_layers == 3
    model = GNN_Multi_Layer(10, 64, 10, num_layers=3)
    assert len(model.convs) == 3
    assert type(model.convs[1]) is GCNConv
    assert model.convs[1].in_channels == 64 and model.convs[1].out_channels == 64