# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import CitationFull

from examples.utils import get_dataset_dir
from torchcp.classification.score import APS
from torchcp.graph.predictor import SplitPredictor
from torchcp.graph.score import DAPS, SNAPS


def build_transductive_gnn_data(data_name, ntrain_per_class=20):
    data_dir = get_dataset_dir()

    if data_name in ['cora_ml']:
        graph_data = CitationFull(data_dir, data_name)[0]

        #######################################
        # training/validation/test data random split
        # ntrain_per_class per class for training/validation, left for test
        #######################################

        classes_idx_set = [(graph_data.y == cls_val).nonzero(
            as_tuple=True)[0] for cls_val in graph_data.y.unique()]
        shuffled_classes = [
            s[torch.randperm(s.shape[0])] for s in classes_idx_set]

        train_idx = torch.concat([s[: ntrain_per_class]
                                  for s in shuffled_classes])
        val_idx = torch.concat(
            [s[ntrain_per_class: 2 * ntrain_per_class] for s in shuffled_classes])
        test_idx = torch.concat([s[2 * ntrain_per_class:]
                                 for s in shuffled_classes])
    else:
        raise NotImplementedError(
            f"The dataset {data_name} has not been implemented!")

    return graph_data, train_idx, val_idx, test_idx


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
    else:
        raise NotImplementedError(
            f"The model {model_name} has not been implemented!")


def train(model, optimizer, graph_data, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    training_loss = F.cross_entropy(out[train_idx], graph_data.y[train_idx])
    training_loss.backward()
    optimizer.step()


def test(model, graph_data, test_idx):
    model.eval()
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index)
        y_pred = torch.argmax(logits, dim=1)
        accuracy = (y_pred[test_idx] == graph_data.y[test_idx]).float().mean().item()
        print(f"Model Acc: {accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    set_seed(seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    data_name = 'cora_ml'
    model_name = 'GCN'

    graph_data, train_idx, val_idx, test_idx = build_transductive_gnn_data(
        data_name)
    graph_data = graph_data.to(device)

    model = build_gnn_model(model_name)(
        graph_data.x.shape[1], 64, graph_data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.001)

    #######################################
    # Training and testing the model
    #######################################

    n_epochs = 200
    for _ in range(n_epochs):
        train(model, optimizer, graph_data, train_idx)

    test(model, graph_data, test_idx)

    #######################################
    # Split Conformal Prediction
    #######################################

    # Note: You can choose other score function, such as:
    # 1. DAPS(graph_data=graph_data,
    #         base_score_function=APS(score_type="softmax"),
    #         neigh_coef=0.5)
    # 2. SNAPS(graph_data=graph_data,
    #          base_score_function=APS(score_type="softmax"),
    #          xi=1 / 3, mu=1 / 3,
    #          features=graph_data.x, k=20)
    score_function = APS(score_type="softmax")

    # split data into calib/evaluration data
    n_calib = 500
    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]

    # calibrate and evaluate with split conformal prediction
    predictor = SplitPredictor(graph_data, score_function, model)
    predictor.calibrate(cal_idx, args.alpha)
    print(score_function.__class__.__name__, predictor.evaluate(eval_idx))