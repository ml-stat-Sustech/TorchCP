# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import pickle

import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import CitationFull, Amazon


from torchcp.classification.score import APS
from torchcp.graph.score import DAPS, SNAPS
from torchcp.graph.predictor import GraphSplitPredictor, NAPSSplitPredictor
from torchcp.graph.utils.metrics import Metrics
from transformers import set_seed


from .utils import *

dataset_dir = get_dataset_dir()
model_dir = get_model_dir()


def test_transductive_graph():
    set_seed(seed=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'cora_ml'
    ntrain_per_class = 20

    if dataset_name in ['cora_ml']:

        dataset = CitationFull(dataset_dir, dataset_name)[0].to(device)
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
    with torch.no_grad():
        logits = best_model(dataset.x, dataset.edge_index)
    y_pred = logits.argmax(dim=1).detach()

    test_accuracy = (y_pred[test_idx] == dataset.y[test_idx]
                     ).sum().item() / test_idx.shape[0]
    print(f"Model Accuracy: {test_accuracy}")

    #######################################
    # The construction of k-NN similarity graph
    #######################################

    knn_edge, knn_weight = compute_adj_knn(dataset.x, k=20)

    #######################################
    # A standard process of split conformal prediction
    #######################################

    alpha = 0.1
    n_calib = 500

    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]

    score_functions = [APS(score_type="softmax"),
                       DAPS(neigh_coef=0.5,
                            base_score_function=APS(score_type="softmax"),
                            graph_data=dataset),
                       SNAPS(lambda_val=1 / 3,
                             mu_val=1 / 3,
                             base_score_function=APS(score_type="softmax"),
                             graph_data=dataset,
                             knn_edge=knn_edge,
                             knn_weight=knn_weight)]

    for score_function in score_functions:
        predictor = GraphSplitPredictor(score_function)
        predictor.calculate_threshold(logits, cal_idx, label_mask, alpha)

        print(
            f"Experiment--Data : {dataset_name}, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
        prediction_sets = predictor.predict_with_logits(logits, eval_idx)

        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(
            f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, dataset.y[eval_idx])}.")
        print(
            f"Average_size: {metrics('average_size')(prediction_sets, dataset.y[eval_idx])}.")
        print(
            f"Singleton_Hit_Ratio: {metrics('singleton_hit_ratio')(prediction_sets, dataset.y[eval_idx])}.")


def test_inductive_graph():
    set_seed(seed=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'Computers'
    model_name = 'GraphSAGE'

    dataset = Amazon(dataset_dir, dataset_name,
                     pre_transform=RandomNodeSplit(split='train_rest', num_val=1000, num_test=10000))
    data = dataset[0].to(device)

    fname = os.path.join(dataset_dir,'Computers_logits.pkl')
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            logits = pickle.load(handle)
    else:
        kwargs = {'batch_size': 512, 'num_workers': 6,
                  'persistent_workers': True}
        train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                      num_neighbors=[25, 10], shuffle=True, **kwargs)
        subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                         num_neighbors=[-1], shuffle=False, **kwargs)

        del subgraph_loader.data.x, subgraph_loader.data.y
        subgraph_loader.data.num_nodes = data.num_nodes
        subgraph_loader.data.n_id = torch.arange(data.num_nodes).to(device)

        hidden_channels = 64
        learning_rate = 0.01

        model = SAGE(dataset.num_features, hidden_channels,
                     dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #######################################
        # Training the model
        #######################################

        n_epochs = 25
        max_val_acc = 0.
        patience = 5
        bad_counter = 0
        best_model = None

        for _ in range(n_epochs):
            model.train()

            for batch in train_loader:
                optimizer.zero_grad()
                y = batch.y[:batch.batch_size]
                y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()

            model.eval()
            y_pred = model.inference(data.x, subgraph_loader).argmax(dim=-1)
            val_acc = int((y_pred[data.val_mask] == data.y[data.val_mask]).sum(
            )) / int(data.val_mask.sum())

            if val_acc >= max_val_acc:
                max_val_acc = val_acc
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
        with torch.no_grad():
            logits = best_model.inference(data.x, subgraph_loader)
            with open(fname, 'wb') as handle:
                pickle.dump(logits, handle, protocol=pickle.HIGHEST_PROTOCOL)
        y_pred = logits.argmax(dim=-1)

        test_accuracy = int((y_pred[data.test_mask] == data.y[data.test_mask]
                             ).sum()) / int(data.test_mask.sum())
        print(f"Model Accuracy: {test_accuracy}")

    #######################################
    # conformal prediction for inductive setting
    #######################################

    alpha = 0.1

    labels = data.y[data.test_mask]
    label_mask = F.one_hot(dataset.y).bool().to(device)[dataset.test_mask]

    logits = logits[data.test_mask]

    metrics = Metrics()
    #######################################
    # basic conformal prediction for inductive setting
    #######################################

    n_calib = 500
    test_idx = torch.arange(dataset.test_mask.sum())
    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]

    score_function = APS(score_type="softmax")

    predictor = GraphSplitPredictor(score_function)
    predictor.calculate_threshold(logits, cal_idx, label_mask, alpha)

    print(
        f"Experiment--Data : {dataset_name}, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
    prediction_sets = predictor.predict_with_logits(logits, eval_idx)


    print("Evaluating prediction sets...")
    print(
        f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, dataset.y[dataset.test_mask][eval_idx])}.")
    print(
        f"Average_size: {metrics('average_size')(prediction_sets, dataset.y[dataset.test_mask][eval_idx])}.")
    print(
        f"Singleton_Hit_Ratio: {metrics('singleton_hit_ratio')(prediction_sets, dataset.y[dataset.test_mask][eval_idx])}.")

    #######################################
    # Neighbourhood Adaptive Prediction Sets for inductive setting
    #######################################

    schemes = ["unif", "linear", "geom"]

    for scheme in schemes:
        predictor = NAPSSplitPredictor(data, scheme=scheme)
        lcc_nodes, prediction_sets = predictor.precompute_naps_sets(
            logits, labels, alpha)

        print(
            f"Experiment--Data : {dataset_name}, Model : {model_name}, Predictor : {predictor.__class__.__name__}, Scheme : {scheme}, Alpha : {alpha}")

        print("Evaluating prediction sets...")
        print(
            f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, labels[lcc_nodes])}.")
        print(
            f"Average_size: {metrics('average_size')(prediction_sets, labels[lcc_nodes])}.")
        print(
            f"Singleton_Hit_Ratio: {metrics('singleton_hit_ratio')(prediction_sets, labels[lcc_nodes])}.")
