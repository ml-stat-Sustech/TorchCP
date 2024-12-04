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
    """
    Args:
        in_channels (int): The number of input feature dimensions.
        hidden_channels (int): The number of hidden feature dimensions.
        out_channels (int): The number of output feature dimensions.
        backbone (str): The GNN model type ('GCN', 'GAT', 'GraphSAGE', 'SGC').
        heads (int): The number of attention heads in GATConv.
        aggr (str): The aggregation method ('sum' or 'mean') for GraphSAGE.
        num_layers (int): The number of layers in the network.
        p_droput (float): The dropout probability.
    """

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
        """
        Forward pass.

        Args:
            x (Tensor): The input feature matrix, shape (num_nodes, in_channels).
            edge_index (Tensor): The edge index, shape (2, num_edges).
            edge_weight (Tensor, optional): The edge weights, shape (num_edges,).

        Returns:
            x (Tensor): The output logits, shape (num_nodes, out_channels).
        """
        for idx, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            if idx == len(self.convs) - 1:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index, edge_weight).relu()
        return x


class ConfGNN(torch.nn.Module):
    """
    Method: Conformalized GNN
    Paper: Uncertainty Quantification over Graph with Conformalized Graph Neural Networks (Huang et al., 2023).
    Link: https://openreview.net/pdf?id=ygjQCOyNfh
    Github: https://github.com/snap-stanford/conformalized-gnn

    Args:
        base_model (str): The type of the base GNN model ('GCN', 'GAT', 'GraphSAGE', 'SGC').
        output_dim (int): The output dimension.
        confnn_hidden_dim (int): The hidden dimension for the Conformal GNN layers.
        num_conf_layers (int): The number of layers in the Conformal GNN.
    """
    def __init__(self, base_model, output_dim, confnn_hidden_dim, num_conf_layers=1):
        super().__init__()
        self.confgnn = GNN_Multi_Layer(
            output_dim, confnn_hidden_dim, output_dim, base_model, num_layers=num_conf_layers)

    def forward(self, logits, edge_index):
        """
        Forward pass for the Conformal GNN.

        Args:
            logits (Tensor): The output logits from the base GNN model, shape (num_nodes, num_classes).
            edge_index (Tensor): The graph edge index, shape (2, num_edges).

        Returns:
            adjust_logits (Tensor): The adjusted logits after applying conformal prediction.
        """
        out = F.softmax(logits, dim=1)
        adjust_logits = self.confgnn(out, edge_index)
        return adjust_logits
