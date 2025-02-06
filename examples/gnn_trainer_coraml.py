# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv
from transformers import set_seed

from examples.utils import get_dataset_dir
from torchcp.classification.score import APS
from torchcp.graph.predictor import SplitPredictor
from torchcp.graph.trainer import CFGNNTrainer


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


def build_transductive_gnn_data(data_name, ntrain_per_class=20):
    data_dir = get_dataset_dir()

    graph_data = CitationFull(data_dir, data_name)[0]

    #######################################
    # training/validation/test data random split
    # ntrain_per_class per class for training/validation, left for test
    #######################################

    classes_idx_set = [(graph_data.y == cls_val).nonzero(
        as_tuple=True)[0] for cls_val in graph_data.y.unique()]
    shuffled_classes = [
        s[torch.randperm(s.shape[0])] for s in classes_idx_set]

    train_idx = torch.concat(
        [s[: ntrain_per_class] for s in shuffled_classes])
    val_idx = torch.concat(
        [s[ntrain_per_class: 2 * ntrain_per_class] for s in shuffled_classes])
    test_idx = torch.concat(
        [s[2 * ntrain_per_class:] for s in shuffled_classes])

    return graph_data, train_idx, val_idx, test_idx


def train(model, optimizer, graph_data, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    training_loss = F.cross_entropy(out[train_idx], graph_data.y[train_idx])
    training_loss.backward()
    optimizer.step()


if __name__ == '__main__':
    set_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    graph_data, train_idx, val_idx, test_idx = build_transductive_gnn_data('cora_ml')
    graph_data = graph_data.to(device)

    model = GCN(in_channels=graph_data.x.shape[1],
                hidden_channels=64,
                out_channels=graph_data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.001)

    #######################################
    # Training and testing the model
    #######################################

    for _ in range(200):
        train(model, optimizer, graph_data, train_idx)

    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)
    pre_logits = F.softmax(logits, dim=1)

    #######################################
    # Split calib/test sets
    #######################################

    rand_perms = torch.randperm(test_idx.size(0))
    calib_train_idx = test_idx[rand_perms[:500]]
    calib_eval_idx = test_idx[rand_perms[500:]]

    #######################################
    # Results with Conformalized GNN
    #######################################

    # Split data into train/val/calib for training
    graph_data['train_idx'] = train_idx
    graph_data['val_idx'] = test_idx
    graph_data['calib_train_idx'] = calib_train_idx
    cf_trainer = CFGNNTrainer(graph_data, model)

    # Train conformalized gnn
    model = cf_trainer.train()
    # breakpoint()

    # Split data into calib/test for evaluating
    eval_perms = torch.randperm(calib_eval_idx.size(0))
    eval_calib_idx = calib_eval_idx[eval_perms[:500]]
    eval_test_idx = calib_eval_idx[eval_perms[500:]]

    #######################################
    # A standard process of conformal prediction
    #######################################
    predictor = SplitPredictor(graph_data,
                               APS(score_type="softmax"),
                               model=model)
    predictor.calibrate(eval_calib_idx, alpha=0.1)

    predict_sets = predictor.predict(eval_test_idx)
    print(predict_sets)

    #########################################
    # Evaluating the coverage rate and average set size on a given dataset.
    ########################################
    result_dict = predictor.evaluate(eval_test_idx)
    print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
    print(f"Average Set Size: {result_dict['average_size']:.4f}")
    print(f"Singleton Hit Ratio: {result_dict['singleton_hit_ratio']:.4f}")
