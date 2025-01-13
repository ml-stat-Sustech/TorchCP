# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from torchcp.classification.loss import ConfTr
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR, APS


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
        backbone_model (torch.nn.Module): backbone model.
        graph_data (from torch_geometric.data import Data): 
            x (tensor): features of nodes.
            edge_index (Tensor): The edge index, shape (2, num_edges).
            edge_weight (Tensor, optional): The edge weights, shape (num_edges,).
            train_idx: The indices of the training nodes.
            val_idx: The indices of the validation nodes.
            calib_train_idx: The indices of the training nodes for CF-GNN.
        hidden_channels (int): Number of hidden channels for the CF-GNN layers.
        alpha (float, optional): The significance level for conformal prediction. Default is 0.1.
    """

    def __init__(
            self,
            base_model,
            graph_data,
            hidden_channels=64,
            alpha=0.1):
        if base_model is None:
            raise ValueError("backbone_model cannot be None.")
        if graph_data is None:
            raise ValueError("graph_data cannot be None.")

        self.base_model = base_model
        self.graph_data = graph_data
        self._device = self.graph_data.x.device

        num_classes = graph_data.y.max().item() + 1
        self.model = GNN_Multi_Layer(in_channels=num_classes,
                                     hidden_channels=hidden_channels,
                                     out_channels=num_classes).to(self._device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=5e-4, lr=0.001)
        self.pred_loss_fn = F.cross_entropy
        self.cf_loss_fn = ConfTr(predictor=SplitPredictor(score_function=THR(score_type="softmax")),
                                 alpha=alpha,
                                 fraction=0.5,
                                 loss_type="classification",
                                 target_size=0)
        self.predictor = SplitPredictor(APS(score_type="softmax"))
        self.alpha = alpha

    def train_epoch(self, epoch, pre_logits):
        """
        Trains the model for one epoch using the given data.

        Args:
            epoch: The current epoch number.
            pre_logits: The preprocessed logits from backbone model.
        """

        self.model.train()
        self.optimizer.zero_grad()

        adjust_logits = self.model(pre_logits, self.graph_data.edge_index)
        loss = self.pred_loss_fn(adjust_logits[self.graph_data.train_idx], self.graph_data.y[self.graph_data.train_idx])

        if epoch >= 1000:
            eff_loss = self.cf_loss_fn(adjust_logits[self.graph_data.calib_train_idx],
                                       self.graph_data.y[self.graph_data.calib_train_idx])
            loss += eff_loss

        loss.backward()
        self.optimizer.step()

    def validate(self, pre_logits):
        """
        Evaluates the model's performance on the validation set.

        Args:
            pre_logits: The preprocessed logits from backbone model.

        Returns:
            eff_valid (float): The average size of validation size.
            adjust_logits: The adjusted logits of CF-GNN.
        """
        self.model.eval()
        with torch.no_grad():
            adjust_logits = self.model(pre_logits, self.graph_data.edge_index)

        size_list = []
        for _ in range(10):
            val_perms = torch.randperm(self.graph_data.val_idx.size(0))
            valid_calib_idx = self.graph_data.val_idx[val_perms[:int(len(self.graph_data.val_idx) / 2)]]
            valid_test_idx = self.graph_data.val_idx[val_perms[int(len(self.graph_data.val_idx) / 2):]]

            self.predictor.calculate_threshold(
                adjust_logits[valid_calib_idx], self.graph_data.y[valid_calib_idx], self.alpha)
            pred_sets = self.predictor.predict_with_logits(
                adjust_logits[valid_test_idx])
            size = self.predictor._metric('average_size')(
                pred_sets, self.graph_data.y[valid_test_idx])
            size_list.append(size)

        return torch.mean(torch.tensor(size_list)), adjust_logits

    def train(self, n_epochs=5000):
        """
        Trains the CF-GNN model for a specified number of epochs and returns the corrected logits.

        Args:
            n_epochs: The number of training epochs.

        Returns:
            model: The model of CF-GNN.
        """
        self.base_model.eval()
        with torch.no_grad():
            logits = self.base_model(self.graph_data.x, self.graph_data.edge_index)
        pre_logits = F.softmax(logits, dim=1)

        best_valid_size = pre_logits.shape[1]

        best_model_dict = None
        for epoch in tqdm(range(n_epochs)):
            self.train_epoch(epoch, pre_logits)

            eff_valid, adjust_logits = self.validate(pre_logits)

            if eff_valid < best_valid_size:
                best_valid_size = eff_valid
                best_model_dict = self.model.state_dict()

        if best_model_dict is not None:
            self.model.load_state_dict(best_model_dict)

        return self.model


class GNN_Multi_Layer(nn.Module):
    """
    Args:
        in_channels (int): The number of input feature dimensions.
        hidden_channels (int): The number of hidden feature dimensions.
        out_channels (int): The number of output feature dimensions.
        p_droput (float): The dropout probability.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, p_droput=0.5):
        super().__init__()
        self.p_dropout = p_droput

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=True, normalize=True))
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
