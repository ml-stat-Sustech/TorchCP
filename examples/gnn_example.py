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

from torchcp.classification.score import APS
from torchcp.graph.score import DAPS
from torchcp.graph.predictor import GraphSplitPredictor, NAPSPredictor
from transformers import set_seed

from torchcp.graph.utils.metrics import Metrics
from examples.utils import build_transductive_gnn_data, build_inductive_gnn_data, build_gnn_model


def train_transductive(model, optimizer, graph_data, train_idx):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    training_loss = F.cross_entropy(out[train_idx], graph_data.y[train_idx])
    training_loss.backward()
    optimizer.step()


def train_inductive(model, optimizer, train_loader):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    set_seed(seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model for transductive
    #######################################

    data_name = 'cora_ml'
    graph_data, label_mask, train_idx, val_idx, test_idx = build_transductive_gnn_data(
        data_name)
    graph_data = graph_data.to(device)

    model = build_gnn_model('GCN')(
        graph_data.x.shape[1], 64, graph_data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.001)

    n_epochs = 200
    #######################################
    # Split Conformal Prediction
    #######################################
    print("########################## CP for Transductive ###########################")
    for _ in range(n_epochs):
        train_transductive(model, optimizer, graph_data, train_idx)

    model.eval()
    score_function = DAPS(neigh_coef=0.5,
                          base_score_function=APS(score_type="softmax"),
                          graph_data=graph_data)
    predictor = GraphSplitPredictor(graph_data, score_function, model)

    n_calib = 500
    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]
    predictor.calibrate(cal_idx, args.alpha)
    print(predictor.evaluate(eval_idx))

    #######################################
    # Loading dataset and a model for inductive
    #######################################

    data_name = 'Computers'
    graph_data, train_loader, subgraph_loader = build_inductive_gnn_data(
        data_name)

    model = build_gnn_model('SAGE')(
        graph_data.x.shape[1], 64, graph_data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 30
    print("########################## CP for Inductive ###########################")
    for _ in range(n_epochs):
        train_inductive(model, optimizer, train_loader)

    model.eval()
    with torch.no_grad():
        logits = model.inference(graph_data.x, subgraph_loader)

    labels = graph_data.y[graph_data.test_mask]
    logits = logits[graph_data.test_mask]

    predictor = NAPSPredictor(graph_data)
    lcc_nodes, prediction_sets = predictor.precompute_naps_sets(
        logits, labels, args.alpha)

    metrics = Metrics()
    print("Evaluating prediction sets...")
    print(
        f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, labels[lcc_nodes])}.")
    print(
        f"Average_size: {metrics('average_size')(prediction_sets, labels[lcc_nodes])}.")
    print(
        f"Singleton_hit_ratio: {metrics('singleton_hit_ratio')(prediction_sets, labels[lcc_nodes])}.")