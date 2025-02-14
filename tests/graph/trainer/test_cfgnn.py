# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from torchcp.classification.score import THR, APS
from torchcp.classification.loss import ConfTrLoss
from torchcp.classification.predictor import SplitPredictor
from torchcp.graph.trainer import CFGNNTrainer
from torchcp.graph.trainer.model import CFGNNModel


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def mock_graph_data(device):
    num_nodes = 200
    x = torch.randn(num_nodes, 10)

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0]
    ])
    y = torch.randint(0, 10, (num_nodes,))
    rand_perm = torch.randperm(num_nodes)
    train_idx = rand_perm[:20]
    val_idx = rand_perm[20:40]
    calib_train_idx = rand_perm[40:60]

    return Data(x=x, edge_index=edge_index, y=y,
                num_nodes=num_nodes,
                train_idx=train_idx,
                val_idx=val_idx,
                calib_train_idx=calib_train_idx).to(device)


@pytest.fixture
def mock_model(device):
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, edge_index=None):
            return x

    return MockModel().to(device)


@pytest.fixture
def mock_cfgnn_model(mock_graph_data, mock_model):
    return CFGNNTrainer(mock_graph_data, mock_model)


def test_initialization(mock_graph_data, mock_model):
    cf_trainer = CFGNNTrainer(mock_graph_data, mock_model)

    assert cf_trainer.graph_data is mock_graph_data
    assert cf_trainer._device == mock_graph_data.x.device
    assert cf_trainer.num_classes == 10

    assert isinstance(cf_trainer.model, CFGNNModel)
    assert cf_trainer.model.base_model is mock_model
    assert cf_trainer.model.model.convs[0].in_channels == 10
    assert cf_trainer.model.model.convs[0].out_channels == 64
    assert len(cf_trainer.model.model.convs) == 2
    assert cf_trainer.model.model.convs[1].in_channels == 64
    assert cf_trainer.model.model.convs[1].out_channels == 10

    assert isinstance(cf_trainer.optimizer, torch.optim.Adam)
    assert cf_trainer.optimizer.param_groups[0]['weight_decay'] == 5e-4
    assert cf_trainer.optimizer.param_groups[0]['lr'] == 0.001

    assert isinstance(cf_trainer.loss_fns, list)
    assert cf_trainer.loss_fns[0] == F.cross_entropy
    assert isinstance(cf_trainer.loss_fns[1], ConfTrLoss)
    assert isinstance(cf_trainer.loss_fns[1].predictor, SplitPredictor)
    assert isinstance(cf_trainer.loss_fns[1].predictor.score_function, THR)
    assert cf_trainer.loss_fns[1].alpha == 0.1
    assert cf_trainer.loss_fns[1].fraction == 0.5
    assert cf_trainer.loss_fns[1].loss_type == "classification"
    assert cf_trainer.loss_fns[1].target_size == 0

    assert cf_trainer.loss_weights[0] == 1.0
    assert cf_trainer.loss_weights[1] == 1.0
    assert isinstance(cf_trainer.predictor, SplitPredictor)
    assert isinstance(cf_trainer.predictor.score_function, APS)
    assert cf_trainer.predictor.score_function.score_type == "softmax"
    assert cf_trainer.alpha == 0.1


def test_invalid_initialization(mock_graph_data, mock_model):
    with pytest.raises(ValueError, match="graph_data cannot be None"):
        CFGNNTrainer(None, mock_model)

    with pytest.raises(ValueError, match="model cannot be None"):
        CFGNNTrainer(mock_graph_data, None)


def test_train_each_epoch( mock_cfgnn_model):
    mock_cfgnn_model._train_each_epoch(500)
    mock_cfgnn_model._train_each_epoch(2000)


def test_evaluate(mock_cfgnn_model, mock_graph_data, mock_model):
    torch.manual_seed(42)
    mock_cfgnn_model.model = mock_model
    size = mock_cfgnn_model._evaluate()

    torch.manual_seed(42)
    mock_model.eval()
    with torch.no_grad():
        logits = mock_model(mock_graph_data.x, mock_graph_data.edge_index)

    val_perms = torch.randperm(mock_graph_data.val_idx.size(0))
    valid_calib_idx = mock_graph_data.val_idx[val_perms[:int(len(mock_graph_data.val_idx) / 2)]]
    valid_test_idx = mock_graph_data.val_idx[val_perms[int(len(mock_graph_data.val_idx) / 2):]]

    mock_cfgnn_model.predictor.calculate_threshold(logits[valid_calib_idx], mock_graph_data.y[valid_calib_idx], 0.1)
    pred_sets = mock_cfgnn_model.predictor.predict_with_logits(logits[valid_test_idx])
    except_size = mock_cfgnn_model.predictor._metric('average_size')(pred_sets, mock_graph_data.y[valid_test_idx])
    assert size == except_size


def test_train(mock_cfgnn_model):
    model = mock_cfgnn_model.train(10)
    assert model is mock_cfgnn_model.model
