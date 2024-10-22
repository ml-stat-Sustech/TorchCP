# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import pickle
import pandas as pd

import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils.convert import to_networkx
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import CitationFull, Amazon


from torchcp.classification.scores import APS
from torchcp.graph.scores import DAPS, SNAPS
from torchcp.graph.predictors import GraphSplitPredictor, NAPSSplitPredictor
from torchcp.graph.utils.metrics import Metrics
from torchcp.utils import fix_randomness


from tests.utils import GCN, SAGE, compute_adj_knn


def test_transductive_graph():
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
    fix_randomness(seed=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'Computers'
    model_name = 'GraphSAGE'

    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data/Amazon")
    dataset = Amazon(data_dir, dataset_name,
                     pre_transform=RandomNodeSplit(split='train_rest', num_val=1000, num_test=10000))
    data = dataset[0].to(device)

    fname = '.cache/Computers_probs.pkl'
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            probs = pickle.load(handle)
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
            probs = F.softmax(best_model.inference(
                data.x, subgraph_loader), dim=-1)
            with open(fname, 'wb') as handle:
                pickle.dump(probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        y_pred = probs.argmax(dim=-1)

        test_accuracy = int((y_pred[data.test_mask] == data.y[data.test_mask]
                             ).sum()) / int(data.test_mask.sum())
        print(f"Model Accuracy: {test_accuracy}")

    #######################################
    # conformal prediction for inductive setting
    #######################################

    alpha = 0.1

    test_subgraph = data.subgraph(data.test_mask)
    G = to_networkx(test_subgraph).to_undirected()
    labels = data.y[data.test_mask]
    probs = probs[data.test_mask]

    #######################################
    # basic conformal prediction for inductive setting
    #######################################

    test_idx = torch.arange(data.x.shape[0], device=device)[data.test_mask]
    label_mask = F.one_hot(dataset.y).bool()

    n_calib = 500
    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]

    score_functions = [APS(score_type="softmax")]

    for score_function in score_functions:
        predictor = GraphSplitPredictor(score_function)
        predictor.calculate_threshold(probs, cal_idx, label_mask, alpha)

        print(
            f"Experiment--Data : {dataset_name}, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
        prediction_sets = predictor.predict_with_logits(probs, eval_idx)

        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(
            f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, dataset.y[eval_idx])}.")
        print(
            f"Average_size: {metrics('average_size')(prediction_sets, dataset.y[eval_idx])}.")
        print(
            f"Singleton_Hit_Ratio: {metrics('singleton_hit_ratio')(prediction_sets, dataset.y[eval_idx])}.")

    #######################################
    # Neighbourhood Adaptive Prediction Sets for inductive setting
    #######################################

    schemes = ["unif", "linear", "geom"]
    # schemes = ["unif"]

    for scheme in schemes:
        predictor = NAPSSplitPredictor(G, scheme=scheme)
        lcc_nodes, prediction_sets = predictor.precompute_naps_sets(probs, labels, alpha)

        print(
            f"Experiment--Data : {dataset_name}, Model : {model_name}, Predictor : {predictor.__class__.__name__}, Scheme : {scheme}, Alpha : {alpha}")

        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(
            f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, labels[lcc_nodes])}.")
        print(
            f"Average_size: {metrics('average_size')(prediction_sets, labels[lcc_nodes])}.")