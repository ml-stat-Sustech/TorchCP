# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import CitationFull

from torchcp.utils import fix_randomness

def test_graph():
    fix_randomness(seed=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, p_dropout):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
            self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
            self.__p_dropout = p_dropout

        def forward(self, x, edge_index, edge_weight=None):
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = F.dropout(x, p=self.__p_dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            return x
    
    #######################################
    # Loading dataset and a model
    #######################################

    dataset_name = 'cora_ml'
    ntrain_per_class = 20
    nval_per_class = 20

    if dataset_name in ['cora_ml']:
        usr_dir = os.path.expanduser('~')
        data_dir = os.path.join(usr_dir, "data")
        data = CitationFull(data_dir, dataset_name)
        dataset = data[0].to(device)

        #######################################
        # training/validation/test data split
        #######################################

        classes = dataset.y.unique()
        classes_idx_set = [(dataset.y == cls_val).nonzero(as_tuple=True)[0] for cls_val in classes]
        shuffled_classes = [s[torch.randperm(s.shape[0])] for s in classes_idx_set]
        split_points = [ntrain_per_class for s in shuffled_classes]

        train_idx = torch.concat([s[: split_points[i_s]] for i_s, s in enumerate(shuffled_classes)])
        val_idx = torch.concat([s[split_points[i_s]: 2*s[split_points[i_s]]] for i_s, s in enumerate(shuffled_classes)])
        test_idx = torch.concat([2*s[split_points[i_s]: ] for i_s, s in enumerate(shuffled_classes)])

        new_train_perm = torch.randperm(train_idx.shape[0])
        new_val_perm = torch.randperm(val_idx.shape[0])
        new_test_perm = torch.randperm(test_idx.shape[0])

        train_idx = train_idx[new_train_perm]
        val_idx = val_idx[new_val_perm]
        test_idx = test_idx[new_test_perm]
    else:
        raise NotImplementedError(f"The dataset {dataset_name} has not been implemented!")
    
    in_channels = dataset.x.shape[1]
    hidden_channels = 64
    out_channels = dataset.y.max().item() + 1
    p_dropout = 0.8

    learning_rate = 0.01
    weight_decay = 0.001

    model = GCN(in_channels, hidden_channels, out_channels, p_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #######################################
    # Training the model
    #######################################

    cache_model_path = ".cache/best_model.pt"

    n_epochs = 1000
    min_validation_loss = 100.
    bad_counter = 0
    saved_flag = False
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(dataset.x, dataset.edge_index)
        training_loss = F.cross_entropy(pred[train_idx], dataset.y[train_idx])
        validation_loss = F.cross_entropy(pred[val_idx], dataset.y[val_idx])

        training_loss.backward()
        optimizer.step()

        if validation_loss.detach().item() <= min_validation_loss:
            torch.save(model, cache_model_path)
            min_validation_loss = validation_loss.detach().item()
            saved_flag = True
            bad_counter = 0
        else:
            bad_counter += 1
        
        if bad_counter == 50:
            break

    if saved_flag is False:
        torch.save(model, cache_model_path)
    else:
        model = torch.load(cache_model_path)
    
    #######################################
    # Test the model
    #######################################

    model.eval()
    pred = model(dataset.x, dataset.edge_index)[test_idx]

    accuracy = accuracy_score(
        y_true=dataset.y[test_idx].cpu().numpy(),
        y_pred=pred.cpu().numpy()
    )
    print(accuracy)