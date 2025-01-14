# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class CFGNNModel(nn.Module):
    """
    Model wrapper for CF-GNN in conformal prediction.
    
    Args:
        base_model (nn.Module): Pre-trained model to be calibrated
        num_classes (int): The number of class.
        hidden_channels (int): The number of hidden feature dimensions.
        num_layers (int): The number of layers in the network.
    
    Examples:
        >>> base_model = GCN(in_channels=100, hidden_channels=64, out_channels=7)
        >>> model = CFGNNModel(base_model, num_class=7)
        >>> logits = model(graph_data.x, graph_data.edge_index)
        
    Reference:
        Huang et al. "Uncertainty Quantification over Graph with Conformalized Graph Neural Networks", NeurIPS 2023, https://arxiv.org/abs/2305.14535
    """

    def __init__(self, 
                 base_model: nn.Module,
                 num_classes: int,
                 hidden_channels: int = 64,
                 num_layers: int = 2):
        super().__init__()
        self.base_model = base_model

        self.model = GNN_Multi_Layer(in_channels=num_classes,
                                     hidden_channels=hidden_channels,
                                     out_channels=num_classes,
                                     num_layers=num_layers)

        self.freeze_base_model()

    def parameters(self, recurse: bool = True):
        """parameters in model of GNN_Multi_Layer"""
        return self.model.parameters(recurse=recurse)

    def freeze_base_model(self):
        """Freeze all parameters in base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def is_base_model_frozen(self) -> bool:
        """Check if base model parameters are frozen"""
        return not any(p.requires_grad for p in self.base_model.parameters())
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass with temperature scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            logits of CF-GNN
        """
        with torch.no_grad():  # Ensure no gradients flow through base model
            logits = self.base_model(x, edge_index)
        pre_probs = F.softmax(logits, dim=1)

        return self.model(pre_probs, edge_index)

    def train(self, mode: bool = True):
        """
        Override train method to ensure base_model stays in eval mode
        
        Args:
            mode: boolean to specify training mode
        """
        super().train(mode)  # Set training mode for TemperatureScalingModel
        self.base_model.eval()  # Keep base_model in eval mode
        return self
    

class GNN_Multi_Layer(nn.Module):
    """
    Args:
        in_channels (int): The number of input feature dimensions.
        hidden_channels (int): The number of hidden feature dimensions.
        out_channels (int): The number of output feature dimensions.
        num_layers (int): The number of layers in the network.
        p_droput (float): The dropout probability.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, p_droput=0.5):
        super().__init__()
        self.p_dropout = p_droput

        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                GCNConv(in_channels, out_channels, cached=True, normalize=True))
        else:
            self.convs.append(
                GCNConv(in_channels, hidden_channels, cached=True, normalize=True))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True))
            self.convs.append(
                GCNConv(hidden_channels, out_channels, cached=True, normalize=True))

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass.

        Args:
            x (Tensor): The output logits of backbone model, shape (num_nodes, num_classes).
            edge_index (Tensor): The edge index, shape (2, num_edges).
            edge_weight (Tensor, optional): The edge weights, shape (num_edges,).

        Returns:
            x (Tensor): The corrected logits, shape (num_nodes, num_classes).
        """
        for idx, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            if idx == len(self.convs) - 1:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index, edge_weight).relu()
        return x
