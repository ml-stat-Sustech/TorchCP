# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import math
import pickle

import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import Amazon


from torchcp.classification.scores import APS, THR
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.loss import ConfTr
from torchcp.graph.scores import DAPS, SNAPS
from torchcp.graph.predictors import GraphSplitPredictor, NAPSPredictor
from torchcp.graph.loss import ConfGNN
from torchcp.graph.utils.metrics import Metrics
from transformers import set_seed


from .utils import *

dataset_dir = get_dataset_dir()
model_dir = get_model_dir()


def test_transductive_graph():
    set_seed(seed=1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'cora_ml'
    graph_data, label_mask, train_idx, val_idx, test_idx = build_graph_dataset(
        dataset_name, device)

    model_name = 'GCN'
    model = GCN(in_channels=graph_data.x.shape[1],
                hidden_channels=64,
                out_channels=graph_data.y.max().item() + 1,
                p_dropout=0.8).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.001)

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
        out = model(graph_data.x, graph_data.edge_index)
        training_loss = F.cross_entropy(
            out[train_idx], graph_data.y[train_idx])
        validation_loss = F.cross_entropy(
            out[val_idx], graph_data.y[val_idx]).detach().item()
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
        logits = best_model(graph_data.x, graph_data.edge_index)
    y_pred = logits.argmax(dim=1).detach()

    test_accuracy = (y_pred[test_idx] == graph_data.y[test_idx]
                     ).sum().item() / test_idx.shape[0]
    print(f"Model Accuracy: {test_accuracy}")

    #######################################
    # The construction of k-NN similarity graph
    #######################################

    knn_edge, knn_weight = compute_adj_knn(graph_data.x, k=20)

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
                            graph_data=graph_data),
                       SNAPS(lambda_val=1 / 3,
                             mu_val=1 / 3,
                             base_score_function=APS(score_type="softmax"),
                             graph_data=graph_data,
                             knn_edge=knn_edge,
                             knn_weight=knn_weight)]

    for score_function in score_functions:
        predictor = GraphSplitPredictor(graph_data, score_function)
        predictor.calculate_threshold(logits, cal_idx, label_mask, alpha)

        print(
            f"Experiment--Data : {dataset_name}, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
        prediction_sets = predictor.predict_with_logits(logits, eval_idx)

        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(
            f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, graph_data.y[eval_idx])}.")
        print(
            f"Average_size: {metrics('average_size')(prediction_sets, graph_data.y[eval_idx])}.")
        print(
            f"Singleton_hit_ratio: {metrics('singleton_hit_ratio')(prediction_sets, graph_data.y[eval_idx])}.")


def test_inductive_graph():
    set_seed(seed=0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'Computers'
    model_name = 'GraphSAGE'

    dataset = Amazon(dataset_dir, dataset_name,
                     pre_transform=RandomNodeSplit(split='train_rest', num_val=1000, num_test=10000))
    graph_data = dataset[0].to(device)

    fname = os.path.join(dataset_dir, 'Computers_logits.pkl')
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            logits = pickle.load(handle).to(device)
    else:
        kwargs = {'batch_size': 512, 'num_workers': 6,
                  'persistent_workers': True}
        train_loader = NeighborLoader(graph_data, input_nodes=graph_data.train_mask,
                                      num_neighbors=[25, 10], shuffle=True, **kwargs)
        subgraph_loader = NeighborLoader(copy.copy(graph_data), input_nodes=None,
                                         num_neighbors=[-1], shuffle=False, **kwargs)

        del subgraph_loader.data.x, subgraph_loader.data.y
        subgraph_loader.data.num_nodes = graph_data.num_nodes
        subgraph_loader.data.n_id = torch.arange(
            graph_data.num_nodes).to(device)

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
            y_pred = model.inference(
                graph_data.x, subgraph_loader).argmax(dim=-1)
            val_acc = int((y_pred[graph_data.val_mask] == graph_data.y[graph_data.val_mask]).sum(
            )) / int(graph_data.val_mask.sum())

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
            logits = best_model.inference(graph_data.x, subgraph_loader)
            with open(fname, 'wb') as handle:
                pickle.dump(logits, handle, protocol=pickle.HIGHEST_PROTOCOL)
        y_pred = logits.argmax(dim=-1)

        test_accuracy = int((y_pred[graph_data.test_mask] == graph_data.y[graph_data.test_mask]
                             ).sum()) / int(graph_data.test_mask.sum())
        print(f"Model Accuracy: {test_accuracy}")

    #######################################
    # conformal prediction for inductive setting
    #######################################

    alpha = 0.1

    labels = graph_data.y[graph_data.test_mask]
    label_mask = F.one_hot(graph_data.y).bool().to(device)[
        graph_data.test_mask]

    logits = logits[graph_data.test_mask]

    metrics = Metrics()
    #######################################
    # basic conformal prediction for inductive setting
    #######################################

    n_calib = 500
    test_idx = torch.arange(graph_data.test_mask.sum())
    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]

    score_function = APS(score_type="softmax")

    predictor = GraphSplitPredictor(graph_data, score_function)
    predictor.calculate_threshold(logits, cal_idx, label_mask, alpha)

    print(
        f"Experiment--Data : {dataset_name}, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
    prediction_sets = predictor.predict_with_logits(logits, eval_idx)


    print("Evaluating prediction sets...")
    print(
        f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, graph_data.y[graph_data.test_mask][eval_idx])}.")
    print(
        f"Average_size: {metrics('average_size')(prediction_sets, graph_data.y[graph_data.test_mask][eval_idx])}.")
    print(
        f"Singleton_hit_ratio: {metrics('singleton_hit_ratio')(prediction_sets, graph_data.y[graph_data.test_mask][eval_idx])}.")

    #######################################
    # Neighbourhood Adaptive Prediction Sets for inductive setting
    #######################################

    schemes = ["unif", "linear", "geom"]

    for scheme in schemes:
        predictor = NAPSPredictor(graph_data, scheme=scheme)
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
            f"Singleton_hit_ratio: {metrics('singleton_hit_ratio')(prediction_sets, labels[lcc_nodes])}.")


def test_conformal_training_graph():
    set_seed(seed=1)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'cora_ml'
    graph_data, label_mask, train_idx, val_idx, test_idx = build_graph_dataset(
        dataset_name, device, split_ratio=True)

    model_name = 'GCN'
    model = GCN(in_channels=graph_data.x.shape[1],
                hidden_channels=64,
                out_channels=graph_data.y.max().item() + 1,
                p_dropout=0.8).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.001)

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
        out = model(graph_data.x, graph_data.edge_index)
        training_loss = F.cross_entropy(
            out[train_idx], graph_data.y[train_idx])
        validation_loss = F.cross_entropy(
            out[val_idx], graph_data.y[val_idx]).detach().item()
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
        logits = best_model(graph_data.x, graph_data.edge_index)
    y_pred = logits.argmax(dim=1).detach()

    test_accuracy = (y_pred[test_idx] == graph_data.y[test_idx]
                     ).sum().item() / test_idx.shape[0]
    print(f"Model Accuracy: {test_accuracy}")

    #######################################
    # Base non-conformity scores
    #######################################
    alpha = 0.05
    n_calib = min(1000, int(test_idx.shape[0] / 2))
    perm = torch.randperm(test_idx.shape[0])
    cal_idx = test_idx[perm[: n_calib]]
    eval_idx = test_idx[perm[n_calib:]]

    predictor = SplitPredictor(score_function=APS(score_type="softmax"))
    predictor._device = device
    predictor.calculate_threshold(
        logits[cal_idx], graph_data.y[cal_idx], alpha)
    prediction_sets = predictor.predict_with_logits(logits[eval_idx])
    res_dict = {"Coverage_rate": predictor._metric('coverage_rate')(prediction_sets, graph_data.y[eval_idx]),
                "Average_size": predictor._metric('average_size')(prediction_sets, graph_data.y[eval_idx])}
    print(res_dict)
    # breakpoint()
    #######################################
    # Initialized Parameter for Conformalized GNN
    #######################################
    epochs = 5000

    calib_fraction = 0.5
    calib_num = min(1000, int(test_idx.shape[0] / 2))
    
    best_valid_size = 10000
    best_logits = logits

    confmodel = ConfGNN(base_model='GCN',
                        output_dim=graph_data.y.max().item() + 1, 
                        confnn_hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(
        confmodel.parameters(), weight_decay=5e-4, lr=0.001)
    criterion = ConfTr(weight=1.0,
                       predictor=SplitPredictor(score_function=THR(score_type="softmax")),
                       alpha=alpha,
                       fraction=0.5,
                       loss_type="cfgnn",
                       target_size=0,
                       base_loss_fn=F.cross_entropy)

    predictor = SplitPredictor(APS(score_type="softmax"))
    predictor._device = device
    #######################################
    # Split calib/test sets
    #######################################
    calib_test_idx = test_idx
    rand_perms = torch.randperm(calib_test_idx.size(0))
    calib_train_idx = calib_test_idx[rand_perms[:int(
        calib_num * calib_fraction)]]
    calib_eval_idx = calib_test_idx[rand_perms[int(
        calib_num * calib_fraction):]]
    
    train_calib_idx = calib_train_idx[int(len(calib_train_idx) / 2):]
    train_test_idx = calib_train_idx[:int(len(calib_train_idx) / 2)]

    print('Starting topology-aware conformal correction...')
    for epoch in tqdm(range(1, epochs + 1)):
        confmodel.train()
        optimizer.zero_grad()

        # adjust_softmax = F.softmax(adjust_logits, dim=1)
        # n_temp = len(train_calib_idx)
        # q_level = math.ceil((n_temp + 1) * (1 - alpha)) / n_temp

        # tps_conformal_scores = adjust_softmax[train_calib_idx,
        #                                       graph_data.y[train_calib_idx]]
        # qhat = torch.quantile(tps_conformal_scores, 1 -
        #                       q_level, interpolation='higher')

        # proxy_size = torch.sigmoid(
        #     (adjust_softmax[train_test_idx] - qhat) / 0.1)
        # size_loss = torch.mean(torch.relu(
        #     torch.sum(proxy_size, dim=1) - 0))

        # pred_loss = F.cross_entropy(
        #     adjust_logits[train_idx], graph_data.y[train_idx])

        # if epoch <= 1000:
        #     loss = pred_loss
        # else:
        #     loss = pred_loss + size_loss
        
        if epoch <= 1000:
            loss = F.cross_entropy(
                adjust_logits[train_idx], graph_data.y[train_idx])
        else:
            loss = criterion(adjust_logits[calib_train_idx], graph_data.y[calib_train_idx])

        loss.backward()
        optimizer.step()

        #######################################
        # Validation Stage
        #######################################
        confmodel.eval()
        with torch.no_grad():
            adjust_logits = confmodel(logits, graph_data.edge_index)

        size_list = []
        for _ in range(10):
            val_perms = torch.randperm(val_idx.size(0))
            valid_calib_idx = val_idx[val_perms[:int(len(val_idx) / 2)]]
            valid_test_idx = val_idx[val_perms[int(len(val_idx) / 2):]]

            predictor.calculate_threshold(
                adjust_logits[valid_calib_idx], graph_data.y[valid_calib_idx], alpha)
            pred_sets = predictor.predict_with_logits(
                adjust_logits[valid_test_idx])
            size = predictor._metric('average_size')(
                pred_sets, graph_data.y[valid_test_idx])
            size_list.append(size)

        eff_valid = np.mean(size_list)

        #######################################
        # Early Stop
        #######################################
        if eff_valid < best_valid_size:
            best_valid_size = eff_valid
            best_logits = adjust_logits

    coverage_list = []
    size_list = []
    for _ in range(100):
        eval_perms = torch.randperm(calib_eval_idx.size(0))
        eval_calib_idx = calib_eval_idx[eval_perms[:int(
            calib_num * calib_fraction)]]
        eval_test_idx = calib_eval_idx[eval_perms[int(
            calib_num * calib_fraction):]]

        predictor.calculate_threshold(
            best_logits[eval_calib_idx], graph_data.y[eval_calib_idx], alpha)
        pred_sets = predictor.predict_with_logits(best_logits[eval_test_idx])

        coverage = predictor._metric('coverage_rate')(
            pred_sets, graph_data.y[eval_test_idx])
        size = predictor._metric('average_size')(
            pred_sets, graph_data.y[eval_test_idx])

        coverage_list.append(coverage)
        size_list.append(size)

    print(np.mean(coverage_list), np.mean(size_list))
