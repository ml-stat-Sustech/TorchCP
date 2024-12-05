# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv

from torchcp.classification.score import THR, APS
from torchcp.classification.loss import ConfTr
from torchcp.classification.predictor import SplitPredictor


class CFGNNTrainer:
    """
    Method: Conformalized GNN
    Paper: Uncertainty Quantification over Graph with Conformalized Graph Neural Networks (Huang et al., 2023).
    Link: https://openreview.net/pdf?id=ygjQCOyNfh
    Github: https://github.com/snap-stanford/conformalized-gnn

    A class for training and evaluating a Conformalized GNN (CF-GNN) for graph classification tasks.
    The model uses a Graph Neural Network (GNN) as the backbone and integrates conformal prediction methods 
    for uncertainty quantification and model calibration.

    Args:
        out_channels (int): Number of output channels (class labels).
        hidden_channels (int): Number of hidden channels for the CF-GNN layers.
        device (str, optional): The device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
        alpha (float, optional): The significance level for conformal prediction. Default is 0.1.
        model (torch.nn.Module, optional): A custom GNN model to use. Default is None, meaning a default GNN model will be created.
        backbone (str, optional): The type of GNN backbone to use (e.g., 'GCN'). Default is 'GCN'.
        heads (int, optional): The number of attention heads for GAT. Default is 1.
        aggr (str, optional): The aggregation method for GraphSAGE. Default is 'sum'.
        num_layers (int, optional): Number of GNN layers. Default is 2.
        p_droput (float, optional): Dropout probability for regularization. Default is 0.5.
    """
    def __init__(self, out_channels, 
                 hidden_channels,
                 device='cpu',
                 alpha=0.1, 
                 model=None, 
                 backbone='GCN', 
                 heads=1, 
                 aggr='sum', 
                 num_layers=2, 
                 p_droput=0.5):
        if model is None:
            self.cfgnn = GNN_Multi_Layer(
                out_channels, hidden_channels, out_channels, backbone, heads, aggr, num_layers, p_droput)
        else:
            self.cfgnn = model
        
        self.cfgnn = self.cfgnn.to(device)
        self.optimizer = torch.optim.Adam(
            self.cfgnn.parameters(), weight_decay=5e-4, lr=0.001)
        self.pred_loss_fn = F.cross_entropy
        self.cf_loss_fn = ConfTr(weight=1.0,
                              predictor=SplitPredictor(score_function=THR(score_type="softmax")),
                              alpha=alpha,
                              fraction=0.5,
                              loss_type="cfgnn",
                              target_size=0)
        self.predictor = SplitPredictor(APS(score_type="softmax"))
        self.alpha = alpha
        self._device = device

    def train_each_epoch(self, epoch, pre_logits, labels, edge_index, train_idx, calib_train_idx):
        """
        Trains the model for one epoch using the given data.

        Args:
            epoch: The current epoch number.
            pre_logits: The preprocessed logits from backbone model.
            labels: The true labels of the nodes.
            edge_index: The edge index of the graph.
            train_idx: The indices of the training nodes.
            calib_train_idx: The indices of the training nodes for CF-GNN.
        """
        self.cfgnn.train()
        self.optimizer.zero_grad()

        adjust_logits = self.cfgnn(pre_logits, edge_index)
        loss = self.pred_loss_fn(adjust_logits[train_idx], labels[train_idx])

        if epoch >= 1000:
            eff_loss = self.cf_loss_fn(adjust_logits[calib_train_idx], labels[calib_train_idx])
            loss += eff_loss

        loss.backward()
        self.optimizer.step()

    def evaluate(self, pre_logits, labels, edge_index, val_idx):
        """
        Evaluates the model's performance on the validation set.

        Args:
            pre_logits: The preprocessed logits from backbone model.
            labels: The true labels of the nodes.
            edge_index: The edge index of the graph.
            val_idx: The indices of the validation nodes.

        Returns:
            eff_valid (float): The average size of validation size.
            adjust_logits: The adjusted logits of CF-GNN.
        """
        self.cfgnn.eval()
        with torch.no_grad():
            adjust_logits = self.cfgnn(pre_logits, edge_index)
        
        size_list = []
        for _ in range(10):
            val_perms = torch.randperm(val_idx.size(0))
            valid_calib_idx = val_idx[val_perms[:int(len(val_idx) / 2)]]
            valid_test_idx = val_idx[val_perms[int(len(val_idx) / 2):]]

            self.predictor.calculate_threshold(
                adjust_logits[valid_calib_idx], labels[valid_calib_idx], self.alpha)
            pred_sets = self.predictor.predict_with_logits(
                adjust_logits[valid_test_idx])
            size = self.predictor._metric('average_size')(
                pred_sets, labels[valid_test_idx])
            size_list.append(size)
        
        return torch.mean(torch.tensor(size_list)), adjust_logits

    def train(self, pre_logits, labels, edge_index, train_idx, val_idx, calib_train_idx, n_epochs=5000):
        """
        Trains the CF-GNN model for a specified number of epochs and returns the corrected logits.

        Args:
            n_epochs: The number of training epochs.
            pre_logits: The preprocessed logits from backbone model.
            labels: The true labels of the nodes.
            edge_index: The edge index of the graph.
            train_idx: The indices of the training nodes.
            val_idx: The indices of the validation nodes.
            calib_train_idx: The indices of the training nodes for CF-GNN.

        Returns:
            best_logits: The corrected logits from the training process of CF-GNN.
        """
        best_valid_size = pre_logits.shape[1]
        best_logits = pre_logits

        for epoch in tqdm(range(n_epochs)):
            self.train_each_epoch(epoch, pre_logits, labels, edge_index, train_idx, calib_train_idx)

            eff_valid, adjust_logits = self.evaluate(pre_logits, labels, edge_index, val_idx)

            if eff_valid < best_valid_size:
                best_valid_size = eff_valid
                best_logits = adjust_logits
        
        return best_logits
    

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
        x = F.softmax(x, dim=1) # softmax(pre_model(x))
        for idx, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.p_dropout, training=self.training)
            if idx == len(self.convs) - 1:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index, edge_weight).relu()
        return x