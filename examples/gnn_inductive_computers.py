# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed

from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import Amazon
from torch_geometric.nn import SAGEConv

from examples.utils import get_dataset_dir
from torchcp.graph.predictor import NAPSPredictor
from torchcp.classification import Metrics


def build_inductive_gnn_data(data_name, n_v=1000, n_t=10000, device='cuda:0'):
    data_dir = get_dataset_dir()

    if data_name in ['Computers']:
        graph_data = Amazon(data_dir, data_name,
                            pre_transform=RandomNodeSplit(split='train_rest', num_val=n_v, num_test=n_t))[0].to(device)
        kwargs = {'batch_size': 512, 'num_workers': 6,
                  'persistent_workers': True}
        train_loader = NeighborLoader(graph_data, input_nodes=graph_data.train_mask,
                                      num_neighbors=[25, 10], shuffle=True, **kwargs)
        subgraph_loader = NeighborLoader(copy.copy(graph_data), input_nodes=None,
                                         num_neighbors=[-1], shuffle=False, **kwargs)

        del subgraph_loader.data.x, subgraph_loader.data.y
        subgraph_loader.data.num_nodes = graph_data.num_nodes
        subgraph_loader.data.n_id = torch.arange(graph_data.num_nodes)
    else:
        raise NotImplementedError(
            f"The dataset {data_name} has not been implemented!")

    return graph_data, train_loader, subgraph_loader


def build_gnn_model(model_name):
    if model_name == "SAGE":
        class SAGE(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, p_dropout=0.5):
                super().__init__()
                self.convs = torch.nn.ModuleList()
                self.convs.append(SAGEConv(in_channels, hidden_channels))
                self.convs.append(SAGEConv(hidden_channels, out_channels))

                self._p_dropout = p_dropout

            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                        x = F.dropout(x, p=self._p_dropout, training=self.training)
                return x

            @torch.no_grad()
            def inference(self, x_all, subgraph_loader):
                device = x_all.device

                # Compute representations of nodes layer by layer, using *all*
                # available edges. This leads to faster computation in contrast to
                # immediately computing the final representations of each batch:
                for i, conv in enumerate(self.convs):
                    xs = []
                    for batch in subgraph_loader:
                        x = x_all[batch.n_id]
                        x = conv(x, batch.edge_index)
                        if i < len(self.convs) - 1:
                            x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())
                    x_all = torch.cat(xs, dim=0).to(device)
                return x_all

        return SAGE
    else:
        raise NotImplementedError(
            f"The model {model_name} has not been implemented!")


def train(model, optimizer, train_loader):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()


def test(model, graph_data, subgraph_loader):
    model.eval()
    with torch.no_grad():
        logits = model.inference(graph_data.x, subgraph_loader)
        y_pred = torch.argmax(logits, dim=1)
        accuracy = (y_pred[graph_data.test_mask] == graph_data.y[graph_data.test_mask]).float().mean().item()
        print(f"Model Acc: {accuracy}")
    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    set_seed(seed=args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model for inductive
    #######################################

    data_name = 'Computers'
    model_name = 'SAGE'

    graph_data, train_loader, subgraph_loader = build_inductive_gnn_data(
        data_name)

    model = build_gnn_model(model_name)(
        graph_data.x.shape[1], 64, graph_data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #######################################
    # Training and testing the model
    #######################################

    n_epochs = 30
    for _ in range(n_epochs):
        train(model, optimizer, train_loader)

    logits = test(model, graph_data, subgraph_loader)
    
    #######################################
    # Neighbourhood Adaptive Prediction Sets for Conformal Prediction
    #######################################

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
