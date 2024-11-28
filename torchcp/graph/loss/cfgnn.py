# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv


class GNN_Multi_Layer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, backbone='GCN', heads=1, aggr='sum', num_layers=2, p_droput=0.5):
        super().__init__()
        self.p_dropout = p_droput

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            if backbone == 'GCN':
                self.convs.append(
                    GCNConv(in_channels, out_channels, cached=True, normalize=True))
            elif backbone == 'GAT':
                self.convs.append(GATConv(in_channels, out_channels, heads))
            elif backbone == 'GraphSAGE':
                self.convs.append(SAGEConv(in_channels, out_channels, aggr))
            elif backbone == 'SGC':
                self.convs.append(SGConv(in_channels, out_channels))
        else:
            if backbone == 'GCN':
                self.convs.append(
                    GCNConv(in_channels, hidden_channels, cached=True, normalize=True))
            elif backbone == 'GAT':
                self.convs.append(GATConv(in_channels, hidden_channels, heads))
            elif backbone == 'GraphSAGE':
                self.convs.append(SAGEConv(in_channels, hidden_channels, aggr))
            elif backbone == 'SGC':
                self.convs.append(SGConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if backbone == 'GCN':
                    self.convs.append(
                        GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True))
                elif backbone == 'GAT':
                    self.convs.append(
                        GATConv(hidden_channels, hidden_channels, heads))
                elif backbone == 'GraphSAGE':
                    self.convs.append(
                        SAGEConv(hidden_channels, hidden_channels, aggr))
                elif backbone == 'SGC':
                    self.convs.append(SGConv(hidden_channels, hidden_channels))
            if backbone == 'GCN':
                self.convs.append(
                    GCNConv(hidden_channels, out_channels, cached=True, normalize=True))
            elif backbone == 'GAT':
                self.convs.append(
                    GATConv(hidden_channels, out_channels, heads))
            elif backbone == 'GraphSAGE':
                self.convs.append(
                    SAGEConv(hidden_channels, out_channels, aggr))
            elif backbone == 'SGC':
                self.convs.append(SGConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_weight=None):
        for idx, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            if idx == len(self.convs) - 1:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index, edge_weight).relu()
        return x


class ConfGNN(torch.nn.Module):
    """
    Conformalized GNN (Huang et al., 2023).
    Paper: https://openreview.net/forum?id=ygjQCOyNfh

    """

    def __init__(self, model, base_model, output_dim, confnn_hidden_dim, num_conf_layers=1):
        super().__init__()
        self.model = model
        self.confgnn = GNN_Multi_Layer(
            output_dim, confnn_hidden_dim, output_dim, base_model, num_layers=num_conf_layers)

    def forward(self, x, edge_index):
        with torch.no_grad():
            logits = self.model(x, edge_index)

        out = F.softmax(logits, dim=1)
        adjust_logits = self.confgnn(out, edge_index)
        return adjust_logits, logits
