# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

def compute_adj_knn(features, k=20):
    if features.shape[0] < k:
        raise ValueError(
                "The number of nodes cannot be less than k.")
    features_normalized = features / features.norm(dim=1, keepdim=True)
    sims = torch.mm(features_normalized, features_normalized.t())
    sims[(torch.arange(len(sims)), torch.arange(len(sims)))] = 0

    topk_values, topk_indices = torch.topk(sims, k, dim=1)

    adj_knn = torch.zeros_like(sims).to(features.device)
    rows = torch.arange(sims.shape[0]).unsqueeze(1).to(features.device)
    adj_knn[rows, topk_indices] = topk_values

    knn_edge = torch.nonzero(adj_knn).T
    knn_weights = adj_knn[knn_edge[0, :], knn_edge[1, :]]

    return knn_edge, knn_weights