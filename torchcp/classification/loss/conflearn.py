# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torchsort import soft_rank, soft_sort

REG_STRENGTH = 0.1 # Regularization strength used in soft sorting for smoothness
B = 50 # A parameter controlling the smoothness of the soft indicator function


class UniformMatchingLoss(torch.nn.Module):
    """
    A custom loss function that calculates the discrepancy in the sorting of input tensor x.
    It measures how far off each element is from its ideal position in a sorted sequence.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass function that computes the loss value for the given input tensor x.
        
        Args:
            x: A tensor (usually the model's output) with shape (batch_size,).
                This represents the predicted scores or values for each sample.
        
        Returns:
            out: A scalar loss value representing the inconsistency in element sorting.
        """
        batch_size = len(x)
        if batch_size == 0:
            return 0
        x_sorted = soft_sort(x.unsqueeze(dim=0), regularization_strength=REG_STRENGTH)
        i_seq = torch.arange(1.0, 1.0 + batch_size, device=x.device) / batch_size
        out = torch.max(torch.abs(i_seq - x_sorted))
        return out


class ConfLearnLoss(torch.nn.Module):
    """
    A loss function used for conformalized uncertainty-aware training of deep multi-class classifiers

    Examples:
        >>> conflearn_loss_fn = ConfLearnLoss()
        >>> output = torch.randn(100, 10)
        >>> target = torch.randint(0, 2, (100,))
        >>> Z_batch = torch.randint(0, 2, (100,))
        >>> loss = conflearn_loss_fn(output, target, Z_batch)
        >>> loss.backward()

    Reference:
        Einbinder et al. "Training Uncertainty-Aware Classifiers with Conformalized Deep Learning" (2022), https://arxiv.org/abs/2205.05878
    """
    def __init__(self):
        super(ConfLearnLoss, self).__init__()

        self.layer_prob = torch.nn.Softmax(dim=1)
        self.criterion_scores = UniformMatchingLoss()

    def forward(self, output, target, Z_batch):
        """
        Forward pass of the conformal loss function. The loss is computed by iterating over different groupings in Z_batch,
        applying the conformal loss for each group, and averaging the loss over all groups.

        Args:
            output (torch.Tensor): The model's output logits (predictions before softmax).
            target (torch.Tensor): The ground truth labels.
            Z_batch (torch.Tensor): A tensor indicating groupings for non-conformity.

        Returns:
            torch.Tensor: The computed loss for the given batch.
        """
        device = output.device
        loss_scores = torch.tensor(0.0, device=device)
        Z_groups = torch.unique(Z_batch)
        n_groups = torch.sum(Z_groups > 0)
        for z in Z_groups:
            if z > 0:
                idx_z = torch.where(Z_batch == z)[0]
                loss_scores_z = self.compute_loss(
                    output[idx_z], target[idx_z])
                loss_scores += loss_scores_z
        loss_scores /= n_groups
        return loss_scores

    def compute_loss(self, y_train_pred, y_train_batch):
        """
        Computes the conformal loss for a given batch of predictions and ground truth.

        Args:
            y_train_pred (torch.Tensor): The model's predicted logits for the batch.
            y_train_batch (torch.Tensor): The ground truth labels for the batch.

        Returns:
            torch.Tensor: The conformal loss for the batch.
        """
        train_proba = self.layer_prob(y_train_pred)
        train_scores = self.__compute_scores_diff(
            train_proba, y_train_batch)
        train_loss_scores = self.criterion_scores(train_scores)
        return train_loss_scores
    
    def __compute_scores_diff(self, proba_values, Y_values):
        """
        Computes the non-conformity scores based on the predicted probabilities and the true labels.
        This score measures how different the predicted probabilities are from the actual labels.

        Args:
            proba_values (torch.Tensor): The predicted probabilities for the batch (after softmax).
            Y_values (torch.Tensor): The ground truth labels for the batch.

        Returns:
            torch.Tensor: The computed non-conformity scores for each sample in the batch.
        """
        device = proba_values.device
        n, K = proba_values.shape
        proba_values = proba_values + 1e-6 * \
            torch.rand(proba_values.shape, dtype=float, device=device)
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
            torch.rand(n, dtype=float, device=device)

        return scores_t
    
    def __soft_indicator(self, x, a, b=B):
        """
        Soft indicator function, which is a smoothed version of a step function.
        This is used for soft indexing in the loss computation to smooth out the discrete jumps.

        Args:
            x (torch.Tensor): The tensor of indices to compute the indicator function over.
            a (torch.Tensor): The rank tensor, indicating the position.
            b (float): Regularization strength, controlling the smoothness of the indicator function.

        Returns:
            torch.Tensor: The soft indicator values.
        """
        out = torch.sigmoid(b * (x - a + 0.5)) - (torch.sigmoid(b * (x - a - 0.5)))
        out = out / (torch.sigmoid(torch.tensor(b * 0.5)) - torch.sigmoid(-torch.tensor(b * 0.5)))
        return out

    def __soft_indexing(self, z, rank):
        """
        Soft indexing operation used to calculate weighted sums based on predicted probabilities
        and the ranks of true labels. This smooths the selection of indices during loss computation.

        Args:
            z (torch.Tensor): The cumulative sorted probabilities.
            rank (torch.Tensor): The ranks corresponding to the true labels.

        Returns:
            torch.Tensor: The weighted sum of indexed values, representing the soft loss.
        """
        n = len(rank)
        K = z.shape[1]
        I = torch.tile(torch.arange(K, device=z.device), (n, 1))
        weight = self.__soft_indicator(I.T, rank).T
        weight = weight * z
        return weight.sum(dim=1)

    
