# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torchsort import soft_rank, soft_sort

REG_STRENGTH = 0.1
B = 50


class UniformMatchingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = len(x)
        if batch_size == 0:
            return 0
        x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
        i_seq = torch.arange(1.0, 1.0 + batch_size, device=x.device) / batch_size
        out = torch.max(torch.abs(i_seq - x_sorted))
        return out


class ConfLearnLoss(torch.nn.Module):
    def __init__(self, device, alpha):
        super(ConfLearnLoss, self).__init__()

        self.device = device
        self.alpha = alpha

        self.layer_prob = torch.nn.Softmax(dim=1)
        self.criterion_scores = UniformMatchingLoss()

    def forward(self, output, target, Z_batch):
        loss_scores = torch.tensor(0.0, device=self.device)
        Z_groups = torch.unique(Z_batch)
        n_groups = torch.sum(Z_groups > 0)
        for z in Z_groups:
            if z > 0:
                idx_z = torch.where(Z_batch == z)[0]
                loss_scores_z = self.compute_loss(
                    output[idx_z], target[idx_z], alpha=self.alpha)
                loss_scores += loss_scores_z
        loss_scores /= n_groups
        return loss_scores

    def compute_loss(self, y_train_pred, y_train_batch, alpha):
        train_proba = self.layer_prob(y_train_pred)
        train_scores = self.__compute_scores_diff(
            train_proba, y_train_batch, alpha=alpha)
        train_loss_scores = self.criterion_scores(train_scores)
        return train_loss_scores
    
    def __compute_scores_diff(self, proba_values, Y_values, alpha=0.1):
        n, K = proba_values.shape
        proba_values = proba_values + 1e-6 * \
            torch.rand(proba_values.shape, dtype=float, device=self.device)
        proba_values = proba_values / torch.sum(proba_values, 1)[:, None]
        ranks_array_t = soft_rank(-proba_values,
                                  regularization_strength=REG_STRENGTH) - 1
        prob_sort_t = -soft_sort(-proba_values,
                                 regularization_strength=REG_STRENGTH)
        Z_t = prob_sort_t.cumsum(dim=1)

        ranks_t = torch.gather(
            ranks_array_t, 1, Y_values.reshape(n, 1)).flatten()
        prob_cum_t = self.__soft_indexing(Z_t, ranks_t)
        prob_final_t = self.__soft_indexing(prob_sort_t, ranks_t)
        scores_t = 1.0 - prob_cum_t + prob_final_t * \
            torch.rand(n, dtype=float, device=self.device)

        return scores_t
    
    def __soft_indicator(self, x, a, b=B):
        out = torch.sigmoid(b * (x - a + 0.5)) - (torch.sigmoid(b * (x - a - 0.5)))
        out = out / (torch.sigmoid(torch.tensor(b * 0.5)) - torch.sigmoid(-torch.tensor(b * 0.5)))
        return out

    def __soft_indexing(self, z, rank):
        n = len(rank)
        K = z.shape[1]
        I = torch.tile(torch.arange(K, device=z.device), (n, 1))
        weight = self.__soft_indicator(I.T, rank).T
        weight = weight * z
        return weight.sum(dim=1)

    
