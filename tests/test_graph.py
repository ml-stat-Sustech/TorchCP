# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from torch_geometric.datasets import CitationFull

from torchcp.classification.scores import APS
from torchcp.graph.scores import DAPS, SNAPS
from torchcp.graph.predictors import GraphSplitPredictor
from torchcp.graph.utils.metrics import Metrics
from torchcp.utils import fix_randomness

from utils import GCN, compute_adj_knn


def test_graph():
    fix_randomness(seed=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'cora_ml'
    ntrain_per_class = 20

    if dataset_name in ['cora_ml']:
        usr_dir = os.path.expanduser('~')
        data_dir = os.path.join(usr_dir, "data")
        dataset = CitationFull(data_dir, dataset_name)[0].to(device)
        label_mask = F.one_hot(dataset.y).bool()

        #######################################
        # training/validation/test data random split
        # 20 per class for training/validation, left for test
        #######################################

        classes_idx_set = [(dataset.y == cls_val).nonzero(
            as_tuple=True)[0] for cls_val in dataset.y.unique()]
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
            f"The dataset {dataset_name} has not been implemented!")

    in_channels = dataset.x.shape[1]
    hidden_channels = 64
    out_channels = dataset.y.max().item() + 1
    p_dropout = 0.8

    learning_rate = 0.01
    weight_decay = 0.001

    model_name = 'GCN'
    model = GCN(in_channels, hidden_channels,
                out_channels, p_dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #######################################
    # Training the model
    #######################################

    n_epochs = 1000
    min_validation_loss = 100.
    patience = 50
    bad_counter = 0
    best_model = None

    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.x, dataset.edge_index)
        training_loss = F.cross_entropy(out[train_idx], dataset.y[train_idx])
        validation_loss = F.cross_entropy(
            out[val_idx], dataset.y[val_idx]).detach().item()
        training_loss.backward()
        optimizer.step()

        if validation_loss <= min_validation_loss:
            min_validation_loss = validation_loss
            best_model = copy.deepcopy(model)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= patience:
            break

    if best_model is None:
        best_model = copy.deepcopy(model)

    #######################################
    # Testing the model
    #######################################

    best_model.eval()
    logits = best_model(dataset.x, dataset.edge_index)
    y_pred = logits.argmax(dim=1).detach()

    test_accuracy = (y_pred[test_idx] == dataset.y[test_idx]
                     ).sum().item() / test_idx.shape[0]
    print(f"Model Accuracy: {test_accuracy}")

    #######################################
    # The construction of k-NN similarity graph
    #######################################

    adj_knn, knn_weights = compute_adj_knn(dataset.x, k=20)

    #######################################
    # A standard process of split conformal prediction
    #######################################

    alpha = 0.05
    n_calib = 500

    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]

    score_functions = [DAPS(neigh_coef=0.5, base_score_function=APS(score_type="softmax")), 
                       SNAPS(lambda_val=1/3, mu_val=1/3, base_score_function=APS(score_type="softmax"))]

    for score_function in score_functions:
        predictor = GraphSplitPredictor(score_function)
        predictor.calculate_threshold(
        logits, cal_idx, label_mask, alpha, dataset.x.shape[0], dataset.edge_index)

        print(f"Experiment--Data : {dataset_name}, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
        prediction_sets = predictor.predict_with_logits(
        logits, eval_idx, dataset.x.shape[0], dataset.edge_index)

        # print(prediction_sets)
        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, dataset.y[eval_idx])}.")
        print(f"Average_size: {metrics('average_size')(prediction_sets, dataset.y[eval_idx])}.")
