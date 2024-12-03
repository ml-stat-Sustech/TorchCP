import pytest

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv

from torchcp.graph.trainer import ConfGNN


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
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, edge_index=None):
            return x
    return MockModel()


def test_initialization():
    # num_layers == 1
    model = ConfGNN(base_model="GCN", output_dim=10, confnn_hidden_dim=64)
    assert len(model.confgnn.convs) == 1
    assert type(model.confgnn.convs[0]) is GCNConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 10

    model = ConfGNN(base_model="GAT", output_dim=10, confnn_hidden_dim=64)
    assert len(model.confgnn.convs) == 1
    assert type(model.confgnn.convs[0]) is GATConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 10
    assert model.confgnn.convs[0].heads == 1

    model = ConfGNN(base_model="GraphSAGE", output_dim=10, confnn_hidden_dim=64)
    assert len(model.confgnn.convs) == 1
    assert type(model.confgnn.convs[0]) is SAGEConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 10
    assert model.confgnn.convs[0].aggr == "sum"

    model = ConfGNN(base_model="SGC", output_dim=10, confnn_hidden_dim=64)
    assert len(model.confgnn.convs) == 1
    assert type(model.confgnn.convs[0]) is SGConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 10

    # num_layers == 2
    model = ConfGNN(base_model="GCN", output_dim=10, confnn_hidden_dim=64, num_conf_layers=2)
    assert len(model.confgnn.convs) == 2
    assert type(model.confgnn.convs[0]) is GCNConv and type(model.confgnn.convs[1]) is GCNConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 64
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 10

    model = ConfGNN(base_model="GAT", output_dim=10, confnn_hidden_dim=64, num_conf_layers=2)
    assert len(model.confgnn.convs) == 2
    assert type(model.confgnn.convs[0]) is GATConv and type(model.confgnn.convs[1]) is GATConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 64
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 10
    assert model.confgnn.convs[0].heads == 1
    assert model.confgnn.convs[1].heads == 1

    model = ConfGNN(base_model="GraphSAGE", output_dim=10, confnn_hidden_dim=64, num_conf_layers=2)
    assert len(model.confgnn.convs) == 2
    assert type(model.confgnn.convs[0]) is SAGEConv and type(model.confgnn.convs[1]) is SAGEConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 64
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 10
    assert model.confgnn.convs[0].aggr == "sum"
    assert model.confgnn.convs[1].aggr == "sum"

    model = ConfGNN(base_model="SGC", output_dim=10, confnn_hidden_dim=64, num_conf_layers=2)
    assert len(model.confgnn.convs) == 2
    assert type(model.confgnn.convs[0]) is SGConv and type(model.confgnn.convs[1]) is SGConv
    assert model.confgnn.convs[0].in_channels == 10 and model.confgnn.convs[0].out_channels == 64
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 10

    # num_layers == 3
    model = ConfGNN(base_model="GCN", output_dim=10, confnn_hidden_dim=64, num_conf_layers=3)
    assert len(model.confgnn.convs) == 3
    assert type(model.confgnn.convs[1]) is GCNConv
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 64

    model = ConfGNN(base_model="GAT", output_dim=10, confnn_hidden_dim=64, num_conf_layers=3)
    assert len(model.confgnn.convs) == 3
    assert type(model.confgnn.convs[1]) is GATConv
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 64
    assert model.confgnn.convs[1].heads == 1

    model = ConfGNN(base_model="GraphSAGE", output_dim=10, confnn_hidden_dim=64, num_conf_layers=3)
    assert len(model.confgnn.convs) == 3
    assert type(model.confgnn.convs[1]) is SAGEConv
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 64
    assert model.confgnn.convs[1].aggr == "sum"

    model = ConfGNN(base_model="SGC", output_dim=10, confnn_hidden_dim=64, num_conf_layers=3)
    assert len(model.confgnn.convs) == 3
    assert type(model.confgnn.convs[1]) is SGConv
    assert model.confgnn.convs[1].in_channels == 64 and model.confgnn.convs[1].out_channels == 64


def test_cfgnn_forward(mock_graph_data, mock_model):
    logits = mock_model(mock_graph_data.x, mock_graph_data.edge_index)
    model = ConfGNN(base_model="GCN", output_dim=10, confnn_hidden_dim=64, num_conf_layers=3)
    adjust_logits = model(logits, mock_graph_data.edge_index)
    assert adjust_logits.shape == logits.shape