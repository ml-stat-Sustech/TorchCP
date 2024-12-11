# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseScore


class KNN(BaseScore):
    """
    Method: K-Nearest Neighbor non-conformity score
    Paper: Hedging Predictions in Machine Learning (Gammerman et al., 2016).
    Link: https://ieeexplore.ieee.org/document/8129828.
    
    Args:
        features (torch.Tensor): The input features of training data.
        labels (torch.Tensor): The labels of training data.
        num_classes (int): The number of classes.
        k (int, optional): The number of neighbors. Default is 1.
        p (float or str, optional): p value for the p-norm distance to calculate between each vector pair. Default is 2. Optional: float or "cosine".
        batch (int, optional): Batch size for distance calculation. Default is None. Set according to your GPU memory; too large may cause out of memory, too small may be slow.
        
    Examples::
        >>> features = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        >>> labels = torch.tensor([0, 1, 0])
        >>> knn = KNN(features, labels, num_classes=2, k=1, p=2, batch=2)
        >>> test_features = torch.tensor([[1.5, 2.5], [2.5, 3.5]])
        >>> scores = knn(test_features)
        >>> print(scores)
    """

    def __init__(self, features, labels, num_classes, k=1, p=2, batch=None):
        super().__init__()
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be an integer greater than or equal to 1.")

        if not (isinstance(p, (float, int)) and p > 0) and p != "cosine":
            raise ValueError("p must be a positive float or 'cosine'.")

        if not (isinstance(batch, int) and batch > 0) and batch != None:
            raise ValueError("batch must be None or a positive integer.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_features = features.to(self.device)
        self.train_labels = labels.to(self.device)
        self.k = k
        self.p = p
        self.num_classes = num_classes
        self.batch = batch

        if self.p == "cosine":
            def cosine_similarity_custom(A, B):
                with torch.no_grad():
                    norm_A = A.norm(dim=1, keepdim=True)
                    norm_B = B.norm(dim=1, keepdim=True)
                    B_T = B.t()
                    dot_product = torch.mm(A, B_T)
                    cosine_sim = dot_product / (norm_A * norm_B.t())
                return cosine_sim

            self.transform = cosine_similarity_custom
        else:
            self.transform = lambda x1, x2: torch.cdist(x1, x2, p=self.p)

    def __call__(self, features, labels=None):
        """
        Calculate non-conformity scores for features.

        Args:
            features (torch.Tensor): The input features.
            labels (torch.Tensor, optional): The ground truth labels. Default is None.

        Returns:
            torch.Tensor: The non-conformity scores.
        """

        features = features.to(self.device)
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        if self.batch != None:
            distances = []
            dataset = TensorDataset(features)
            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=False)
            for batch_feature in dataloader:
                distances.append(self.transform(batch_feature[0], self.train_features))
                del batch_feature
                torch.cuda.empty_cache()

            distances = torch.cat(distances)
        else:
            distances = self.transform(features, self.train_features)

        if labels is None:
            return self.__calculate_all_label(distances)
        else:
            labels = labels.to(self.device)
            return self.__calculate_single_label(distances, labels)

    def __calculate_single_label(self, distances, labels):
        """
        Calculate non-conformity scores for all labels.

        Args:
            distances (torch.Tensor): The distances between features and training data.

        Returns:
            torch.Tensor: The non-conformity scores.
        """

        labels_expanded = labels.unsqueeze(1).expand(-1, distances.size(1))
        same_label_mask = self.train_labels == labels_expanded
        diff_label_mask = ~same_label_mask

        same_label_distances = torch.where(same_label_mask, distances, torch.full_like(distances, float('inf')))
        diff_label_distances = torch.where(diff_label_mask, distances, torch.full_like(distances, float('inf')))

        topk_same_label_distances, _ = torch.topk(same_label_distances, self.k, largest=False, dim=1)
        topk_diff_label_distances, _ = torch.topk(diff_label_distances, self.k, largest=False, dim=1)

        return torch.sum(topk_same_label_distances, dim=1) / torch.sum(topk_diff_label_distances, dim=1)

    def __calculate_all_label(self, distances):
        """
        Calculate non-conformity score for a single label.

        Args:
            distances (torch.Tensor): The distances between features and training data.
            labels (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The non-conformity score for the given label.
        """
        scores = torch.zeros((distances.shape[0], self.num_classes), device=self.device)
        for label in range(self.num_classes):
            label_tensor = torch.full((distances.shape[0],), label, dtype=torch.long, device=self.device)
            scores[:, label] = self.__calculate_single_label(distances, label_tensor)
        return scores
