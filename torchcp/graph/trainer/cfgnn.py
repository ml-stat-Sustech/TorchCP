# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchcp.classification.loss import ConfTrLoss
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR, APS
from torchcp.graph.trainer.model import CFGNNModel


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
        graph_data (from torch_geometric.data import Data): 
            x (tensor): features of nodes.
            edge_index (Tensor): The edge index, shape (2, num_edges).
            edge_weight (Tensor, optional): The edge weights, shape (num_edges,).
            train_idx: The indices of the training nodes.
            val_idx: The indices of the validation nodes.
            calib_train_idx: The indices of the training nodes for CF-GNN.
        model (torch.nn.Module): backbone model.
        hidden_channels (int): Number of hidden channels for the CF-GNN layers.
        num_layers (int): The number of layers in the network.
        alpha (float, optional): The significance level for conformal prediction. Default is 0.1.
        optimizer_class (torch.optim.Optimizer): Optimizer class for temperature parameter
                Default: torch.optim.Adam
        optimizer_params (dict): Parameters passed to optimizer constructor
                Default: {'weight_decay': 5e-4, 'lr': 0.001}
    """

    def __init__(
            self,
            graph_data,
            model,
            hidden_channels=64,
            num_layers=2,
            alpha=0.1,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_params: dict = {'weight_decay': 5e-4, 'lr': 0.001}):

        if graph_data is None:
            raise ValueError("graph_data cannot be None.")
        if model is None:
            raise ValueError("model cannot be None.")

        self.graph_data = graph_data
        self._device = self.graph_data.x.device

        self.num_classes = graph_data.y.max().item() + 1
        self.model = CFGNNModel(model, self.num_classes, hidden_channels, num_layers).to(self._device)

        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_params
        )

        self.loss_fns = [F.cross_entropy,
                         ConfTrLoss(predictor=SplitPredictor(score_function=THR(score_type="softmax")),
                                alpha=alpha,
                                fraction=0.5,
                                loss_type="classification",
                                target_size=0)]
        self.loss_weights = [1.0, 1.0]

        self.predictor = SplitPredictor(APS(score_type="softmax"))
        self.alpha = alpha

    def _train_each_epoch(self, epoch):
        """
        Trains the model for one epoch using the given data.

        Args:
            epoch: The current epoch number.
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(self.graph_data.x, self.graph_data.edge_index)
        loss = self.loss_fns[0](logits[self.graph_data.train_idx], self.graph_data.y[self.graph_data.train_idx])

        if epoch >= 1000:
            eff_loss = self.loss_fns[1](logits[self.graph_data.calib_train_idx],
                                        self.graph_data.y[self.graph_data.calib_train_idx])
            loss = self.loss_weights[0] * loss + self.loss_weights[1] * eff_loss

        loss.backward()
        self.optimizer.step()

    def _evaluate(self):
        """
        Evaluates the model's performance on the validation set.

        Returns:
            size (float): The average size of validation size.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.graph_data.x, self.graph_data.edge_index)

        val_perms = torch.randperm(self.graph_data.val_idx.size(0))
        valid_calib_idx = self.graph_data.val_idx[val_perms[:int(len(self.graph_data.val_idx) / 2)]]
        valid_test_idx = self.graph_data.val_idx[val_perms[int(len(self.graph_data.val_idx) / 2):]]

        self.predictor.calculate_threshold(logits[valid_calib_idx], self.graph_data.y[valid_calib_idx], self.alpha)
        pred_sets = self.predictor.predict_with_logits(logits[valid_test_idx])
        size = self.predictor._metric('average_size')(pred_sets, self.graph_data.y[valid_test_idx])

        return size

    def train(self, n_epochs=5000):
        """
        Trains the CF-GNN model for a specified number of epochs and returns the corrected logits.

        Args:
            n_epochs: The number of training epochs.

        Returns:
            model: The best model of CF-GNN.
        """

        best_valid_size = self.num_classes
        best_model_state = None

        for epoch in tqdm(range(n_epochs)):
            self._train_each_epoch(epoch)

            eff_valid = self._evaluate()

            if eff_valid < best_valid_size:
                best_valid_size = eff_valid
                best_model_state = copy.deepcopy(self.model.state_dict())

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        return self.model
