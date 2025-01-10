# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import set_seed

from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Amazon
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit

from examples.utils import get_dataset_dir
from torchcp.graph.predictor import NAPSPredictor


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


def build_inductive_gnn_data(data_name, n_v=1000, n_t=10000, device='cuda:0'):
    data_dir = get_dataset_dir()

    graph_data = Amazon(data_dir, data_name,
                        pre_transform=RandomNodeSplit(split='train_rest', 
                                                      num_val=n_v, 
                                                      num_test=n_t))[0].to(device)
    kwargs = {'batch_size': 512, 'num_workers': 6,
              'persistent_workers': True}
    train_loader = NeighborLoader(graph_data, input_nodes=graph_data.train_mask,
                                  num_neighbors=[25, 10], shuffle=True, **kwargs)
    return graph_data, train_loader


def train(model, optimizer, train_loader):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    set_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #######################################
    # Loading dataset and a model for inductive
    #######################################

    graph_data, train_loader = build_inductive_gnn_data('Computers')

    model = SAGE(in_channels=graph_data.x.shape[1],
                 hidden_channels=64,
                 out_channels=graph_data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #######################################
    # Training and testing the model
    #######################################

    for _ in range(30):
        train(model, optimizer, train_loader)

    #######################################
    # Neighbourhood Adaptive Prediction Sets for Conformal Prediction
    #######################################

    eval_idx = torch.where(graph_data.test_mask)[0]

    predictor = NAPSPredictor(graph_data, model=model)
    print(predictor.evaluate(eval_idx, alpha=0.1))
