import pytest

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv

from torchcp.classification.loss import ConfTr
from torchcp.graph.trainer import CFGNNTrainer
from torchcp.graph.trainer.cfgnn import GNN_Multi_Layer
from torchcp.classification.predictor import SplitPredictor


@pytest.fixture
def mock_graph_data():
    num_nodes = 200
    x = torch.randn(num_nodes, 10)

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0]
    ])
    y = torch.randint(0, 10, (num_nodes, ))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


@pytest.fixture
def mock_cfgnn_model():
    return CFGNNTrainer(out_channels=10, hidden_channels=64, device='cuda:0')


@pytest.fixture
def device():
    return torch.device('cuda:0')


def test_initialization():
    # num_layers == 1
    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GCN", num_layers=1)
    assert len(model.cfgnn.convs) == 1
    assert type(model.cfgnn.convs[0]) is GCNConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 10

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GAT", num_layers=1)
    assert len(model.cfgnn.convs) == 1
    assert type(model.cfgnn.convs[0]) is GATConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 10
    assert model.cfgnn.convs[0].heads == 1

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GraphSAGE", num_layers=1)
    assert len(model.cfgnn.convs) == 1
    assert type(model.cfgnn.convs[0]) is SAGEConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 10
    assert model.cfgnn.convs[0].aggr == "sum"

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="SGC", num_layers=1)
    assert len(model.cfgnn.convs) == 1
    assert type(model.cfgnn.convs[0]) is SGConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 10

    # # num_layers == 2
    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GCN", num_layers=2)
    assert len(model.cfgnn.convs) == 2
    assert type(model.cfgnn.convs[0]) is GCNConv and type(model.cfgnn.convs[1]) is GCNConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 64
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 10

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GAT", num_layers=2)
    assert len(model.cfgnn.convs) == 2
    assert type(model.cfgnn.convs[0]) is GATConv and type(model.cfgnn.convs[1]) is GATConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 64
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 10
    assert model.cfgnn.convs[0].heads == 1
    assert model.cfgnn.convs[1].heads == 1

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GraphSAGE", num_layers=2)
    assert len(model.cfgnn.convs) == 2
    assert type(model.cfgnn.convs[0]) is SAGEConv and type(model.cfgnn.convs[1]) is SAGEConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 64
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 10
    assert model.cfgnn.convs[0].aggr == "sum"
    assert model.cfgnn.convs[1].aggr == "sum"

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="SGC", num_layers=2)
    assert len(model.cfgnn.convs) == 2
    assert type(model.cfgnn.convs[0]) is SGConv and type(model.cfgnn.convs[1]) is SGConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 64
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 10

    # # num_layers == 3
    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GCN", num_layers=3)
    assert len(model.cfgnn.convs) == 3
    assert type(model.cfgnn.convs[1]) is GCNConv
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 64

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GAT", num_layers=3)
    assert len(model.cfgnn.convs) == 3
    assert type(model.cfgnn.convs[1]) is GATConv
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 64
    assert model.cfgnn.convs[1].heads == 1

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="GraphSAGE", num_layers=3)
    assert len(model.cfgnn.convs) == 3
    assert type(model.cfgnn.convs[1]) is SAGEConv
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 64
    assert model.cfgnn.convs[1].aggr == "sum"

    model = CFGNNTrainer(out_channels=10, hidden_channels=64, backbone="SGC", num_layers=3)
    assert len(model.cfgnn.convs) == 3
    assert type(model.cfgnn.convs[1]) is SGConv
    assert model.cfgnn.convs[1].in_channels == 64 and model.cfgnn.convs[1].out_channels == 64

    assert next(model.cfgnn.parameters()).device == torch.device(model._device)
    assert type(model.optimizer) is torch.optim.Adam
    assert model.pred_loss_fn == F.cross_entropy
    assert type(model.cf_loss_fn) is ConfTr
    assert type(model.predictor) is SplitPredictor
    assert model.alpha == 0.1

    cfgnn = GNN_Multi_Layer(10, 64, 10, num_layers=1)
    model = CFGNNTrainer(out_channels=10, hidden_channels=64, model=cfgnn)
    assert len(model.cfgnn.convs) == 1
    assert type(model.cfgnn.convs[0]) is GCNConv
    assert model.cfgnn.convs[0].in_channels == 10 and model.cfgnn.convs[0].out_channels == 10


def test_train_each_epoch(mock_graph_data, mock_cfgnn_model, device):
    mock_graph_data = mock_graph_data.to(device)
    rand_perm = torch.randperm(mock_graph_data.x.shape[0]).to(device)
    train_idx = rand_perm[:20]
    calib_train_idx = rand_perm[20:40]

    mock_cfgnn_model.train_each_epoch(500, 
                                      mock_graph_data.x, 
                                      mock_graph_data.y, 
                                      mock_graph_data.edge_index, 
                                      train_idx,
                                      calib_train_idx)
    
    mock_cfgnn_model.train_each_epoch(2000, 
                                      mock_graph_data.x, 
                                      mock_graph_data.y, 
                                      mock_graph_data.edge_index, 
                                      train_idx,
                                      calib_train_idx)
    
def test_evaluate(mock_graph_data, mock_cfgnn_model, device):
    mock_graph_data = mock_graph_data.to(device)
    val_idx = torch.arange(50).to(device)
    results = mock_cfgnn_model.evaluate(mock_graph_data.x,
                                mock_graph_data.y, 
                                mock_graph_data.edge_index,
                                val_idx)
    assert len(results) == 2
    assert results[1].shape == mock_graph_data.x.shape


def test_train(mock_graph_data, mock_cfgnn_model, device):
    mock_graph_data = mock_graph_data.to(device)
    pre_logits = mock_graph_data.x
    labels = mock_graph_data.y
    edge_index = mock_graph_data.edge_index

    rand_perm = torch.randperm(mock_graph_data.x.shape[0]).to(device)
    train_idx = rand_perm[:20]
    val_idx = rand_perm[20:40]
    calib_train_idx = rand_perm[40:60]

    results = mock_cfgnn_model.train(pre_logits, labels, edge_index, train_idx, val_idx, calib_train_idx, n_epochs=10)
    assert results.shape == pre_logits.shape