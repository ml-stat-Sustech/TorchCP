import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv


def build_regression_model(model_name):
    if model_name == "NonLinearNet":
        class NonLinearNet(nn.Module):
            def __init__(self, in_shape, out_shape, hidden_size, dropout):
                super(NonLinearNet, self).__init__()
                self.hidden_size = hidden_size
                self.in_shape = in_shape
                self.out_shape = out_shape
                self.dropout = dropout
                self.base_model = nn.Sequential(
                    nn.Linear(self.in_shape, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.out_shape),
                )

            def forward(self, x):
                return self.base_model(x)

        return NonLinearNet
    else:
        raise NotImplementedError


def build_gnn_model(model_name):
    if model_name == "GCN":
        class GCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, p_dropout=0.8):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
                self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
                self._p_dropout = p_dropout

            def forward(self, x, edge_index, edge_weight=None):
                x = self.conv1(x, edge_index, edge_weight).relu()
                x = F.dropout(x, p=self._p_dropout, training=self.training)
                x = self.conv2(x, edge_index, edge_weight)
                return x
        return GCN
    elif model_name == "SAGE":
        class SAGE(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, p_dropout=0.5):
                super().__init__()
                self.convs = torch.nn.ModuleList()
                self.convs.append(SAGEConv(in_channels, hidden_channels))
                self.convs.append(SAGEConv(hidden_channels, out_channels))
                
                self._p_dropout = p_dropout

            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                        x = F.dropout(x, p=self._p_dropout, training=self.training)
                return x

            @torch.no_grad()
            def inference(self, x_all, subgraph_loader):
                device = x_all.device

                # Compute representations of nodes layer by layer, using *all*
                # available edges. This leads to faster computation in contrast to
                # immediately computing the final representations of each batch:
                for i, conv in enumerate(self.convs):
                    xs = []
                    for batch in subgraph_loader:
                        x = x_all[batch.n_id]
                        x = conv(x, batch.edge_index)
                        if i < len(self.convs) - 1:
                            x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())
                    x_all = torch.cat(xs, dim=0).to(device)
                return x_all
        return SAGE
    else:
        raise NotImplementedError