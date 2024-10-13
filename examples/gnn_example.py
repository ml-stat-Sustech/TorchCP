# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse

import torch
import torch.nn.functional as F

from torchcp.classification.scores import APS
from torchcp.graph.scores import DAPS
from torchcp.graph.predictors import GraphSplitPredictor
from torchcp.utils import fix_randomness

from examples.common.utils import build_gnn_model
from examples.common.dataset import build_gnn_data


def train_transductive(model, optimizer, dataset, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(dataset.x, dataset.edge_index)
    training_loss = F.cross_entropy(out[train_idx], dataset.y[train_idx])
    training_loss.backward()
    optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--data_name', default='cora_ml', type=str)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset, label_mask, train_idx, val_idx, test_idx = build_gnn_data(
        args.data_name)
    dataset = dataset.to(device)

    model = build_gnn_model('GCN')(
        dataset.x.shape[1], 64, dataset.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.001)

    n_epochs = 200
    #######################################
    # Split Conformal Prediction
    #######################################
    print("########################## CP for Transductive ###########################")
    for epoch in range(n_epochs):
        train_transductive(model, optimizer, dataset, train_idx)

    model.eval()
    score_function = DAPS(neigh_coef=0.5,
                          base_score_function=APS(score_type="softmax"),
                          graph_data=dataset)
    predictor = GraphSplitPredictor(score_function, model, dataset)

    n_calib = 500
    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]
    predictor.calibrate(cal_idx, args.alpha)
    print(predictor.evaluate(eval_idx))
