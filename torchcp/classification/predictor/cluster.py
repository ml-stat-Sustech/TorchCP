# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

import numpy as np
import torch
from sklearn.cluster import KMeans

from torchcp.classification.predictor.class_conditional import ClassConditionalPredictor
from torchcp.utils.common import DimensionError


class ClusteredPredictor(ClassConditionalPredictor):
    """
    Method: Clutered Conforml Predictor 
    Paper: Class-Conditional Conformal Prediction with Many Classes (Ding et al., 2023)
    Link: https://arxiv.org/abs/2306.09335
    Github: https://github.com/tiffanyding/class-conditional-conformal
    
    The class implements class-conditional conformal prediction with many classes.

    Args:
        score_function (callable): A non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        alpha (float, optional): The significance level. Default is 0.1.
        ratio_clustering (str or float, optional): The ratio of examples in the calibration dataset used to cluster classes. Default is "auto".
        num_clusters (str or int, optional): The number of clusters. If ratio_clustering is "auto", the number of clusters is automatically computed. Default is "auto".
        split (str, optional): The method to split the dataset into clustering dataset and calibration set. Options are 'proportional', 'doubledip', or 'random'. Default is 'random'.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
        device (torch.device, optional): The device on which the model is located. Default is None.

    Attributes:
        __ratio_clustering (str or float): The ratio of examples in the calibration dataset used to cluster classes.
        __num_clusters (str or int): The number of clusters.
        __split (str): The method to split the dataset into clustering dataset and calibration set.
    """

    def __init__(self, score_function, model=None, temperature=1, alpha=0.1, ratio_clustering="auto", num_clusters="auto",
                 split='random', device=None
                 ):
        super(ClusteredPredictor, self).__init__(score_function, model, temperature, alpha, device)

        if ratio_clustering != "auto" and not (0 < ratio_clustering < 1):
            raise ValueError("ratio_clustering should be 'auto' or a value in (0, 1).")

        if num_clusters != "auto" and not (isinstance(num_clusters, int) and num_clusters > 0):
            raise ValueError("num_clusters should be 'auto' or a positive integer.")

        if split not in ['proportional', 'doubledip', 'random']:
            raise ValueError("split should be one of 'proportional', 'doubledip', or 'random'.")

        self.__ratio_clustering = ratio_clustering
        self.__num_clusters = num_clusters
        self.__split = split

    def calculate_threshold(self, logits, labels, alpha=None):
        """
        Calculate the class-wise conformal prediction thresholds.

        Args:
            logits (torch.Tensor): The logits output from the model.
            labels (torch.Tensor): The ground truth labels.
            alpha (float): The significance level. Default is None.
        """

        if alpha is None:
            alpha = self.alpha
        
        cluster_assignments, cal_scores, cal_labels = self.preprocess_scores(logits, labels, alpha)

        self.q_hat = self.__compute_cluster_specific_qhats(cluster_assignments,cal_scores,cal_labels,alpha)
        
    def preprocess_scores(self, logits, labels, alpha):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        num_classes = logits.shape[1]
        scores = self.score_function(logits, labels)

        alpha = torch.tensor(alpha, device=self._device)
        classes_statistics = torch.tensor([torch.sum(labels == k).item() for k in range(num_classes)],
                                          device=self._device)

        # 1) Choose necessary parameters for Cluster algorithm
        if self.__ratio_clustering == 'auto' or self.__num_clusters == 'auto':
            n_min = torch.min(classes_statistics)
            n_thresh = self.__get_quantile_minimum(alpha)
            # Classes with fewer than n_thresh examples will be excluded from clustering
            n_min = torch.maximum(n_min, n_thresh)
            num_remaining_classes = torch.sum((classes_statistics >= n_min).float())

            # Compute the number of clusters and the minium number of examples for each class
            n_clustering = (n_min * num_remaining_classes / (75 + num_remaining_classes)).clone().to(
                torch.int32).to(self._device)
            if self.__num_clusters == 'auto':
                self.__num_clusters = torch.floor(n_clustering / 2).to(torch.int32)
            if self.__ratio_clustering == 'auto':
                self.__ratio_clustering = n_clustering / n_min

        # 2) Split data
        clustering_scores, clustering_labels, cal_scores, cal_labels = self.__split_data(scores,
                                                                                         labels,
                                                                                         classes_statistics)

        # 3)  Filter "rare" classes
        rare_classes = self.__get_rare_classes(clustering_labels, alpha, num_classes)
        self.num_clusters = self.__num_clusters
        # 4) Run clustering
        if (num_classes - len(rare_classes) > self.__num_clusters) and (self.__num_clusters > 1):
            # Filter out rare classes and re-index
            remaining_idx, filtered_labels, class_remapping = self.__remap_classes(clustering_labels, rare_classes)
            filtered_scores = clustering_scores[remaining_idx]

            # Compute embedding for each class and get class counts
            embeddings, class_cts = self.__embed_all_classes(filtered_scores, filtered_labels)
            kmeans = KMeans(n_clusters=int(self.__num_clusters), n_init=10, random_state=2023).fit(
                X=embeddings.detach().cpu().numpy(),
                sample_weight=np.sqrt(
                    class_cts.detach().cpu().numpy()),
            )
            nonrare_class_cluster_assignments = torch.tensor(kmeans.labels_, device=self._device)

            cluster_assignments = - torch.ones((num_classes,), dtype=torch.int32, device=self._device)

            for cls, remapped_cls in class_remapping.items():
                cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
        else:
            cluster_assignments = - torch.ones((num_classes,), dtype=torch.int32, device=self._device)
        self.cluster_assignments = cluster_assignments
        return cluster_assignments, cal_scores, cal_labels 

    def __split_data(self, scores, labels, classes_statistics):
        """
        Split the data into clustering dataset and calibration set.

        Args:
            scores (torch.Tensor): The non-conformity scores.
            labels (torch.Tensor): The ground truth labels.
            classes_statistics (torch.Tensor): The statistics of each class.

        Returns:
            tuple: A tuple containing clustering scores, clustering labels, calibration scores, and calibration labels.
        """

        if self.__split == 'proportional':
            # Split dataset along with fraction "frac_clustering"
            num_classes = classes_statistics.shape[0]
            n_k = torch.tensor([int(self.__ratio_clustering * classes_statistics[k]) for k in range(num_classes)],
                               device=self._device, dtype=torch.int32)
            idx1 = torch.zeros(labels.shape, dtype=torch.bool, device=self._device)
            for k in range(num_classes):
                # Randomly select n instances of class k
                idx = torch.argwhere(labels == k).flatten()
                random_indices = torch.randint(0, classes_statistics[k], (n_k[k],), device=self._device)
                selected_idx = idx[random_indices]
                idx1[selected_idx] = 1
            clustering_scores = scores[idx1]
            clustering_labels = labels[idx1]
            cal_scores = scores[~idx1]
            cal_labels = labels[~idx1]

        elif self.__split == 'doubledip':
            clustering_scores, clustering_labels = scores, labels
            cal_scores, cal_labels = scores, labels
            idx1 = torch.ones((scores.shape[0])).bool()

        elif self.__split == 'random':
            # Each point is assigned to clustering set w.p. frac_clustering 
            idx1 = torch.rand(size=(len(labels),), device=self._device) < self.__ratio_clustering

            clustering_scores = scores[idx1]
            clustering_labels = labels[idx1]
            cal_scores = scores[~idx1]
            cal_labels = labels[~idx1]

        self.idx1 = idx1
        return clustering_scores, clustering_labels, cal_scores, cal_labels

    def __get_quantile_minimum(self, alpha):
        """
        Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1

        Args:
            alpha (torch.Tensor): The significance level.

        Returns:
            torch.Tensor: The smallest n.
        """
        n = torch.tensor(0, device=alpha.device)
        while torch.ceil((n + 1) * (1 - alpha) / n) > 1:
            n += 1
        return n

    def __get_rare_classes(self, labels, alpha, num_classes):
        """
        Choose classes whose number is less than or equal to the threshold.

        Args:
            labels (torch.Tensor): The ground truth labels.
            alpha (torch.Tensor): The significance level.
            num_classes (int): The number of classes.

        Returns:
            torch.Tensor: The rare classes.
        """
        thresh = self.__get_quantile_minimum(alpha)
        classes, cts = torch.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh].to(self._device)

        # Also included any classes that are so rare that we have 0 labels for it

        all_classes = torch.arange(num_classes, device=self._device)
        zero_ct_classes = all_classes[(all_classes.view(1, -1) != classes.view(-1, 1)).all(dim=0)]
        rare_classes = torch.concatenate((rare_classes, zero_ct_classes))

        return rare_classes

    def __remap_classes(self, labels, rare_classes):
        """
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed.

        Args:
            labels (torch.Tensor): The ground truth labels.
            rare_classes (torch.Tensor): The rare classes.

        Returns:
            tuple: A tuple containing remaining_idx, remapped_labels, and remapping.
        """
        labels = labels.detach().cpu().numpy()
        rare_classes = rare_classes.detach().cpu().numpy()
        remaining_idx = ~np.isin(labels, rare_classes)

        remaining_labels = labels[remaining_idx]
        remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1

        return torch.from_numpy(remaining_idx).to(self._device), torch.tensor(remapped_labels,
                                                                              device=self._device), remapping

    def __embed_all_classes(self, scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
        """
        Embed all classes based on the quantiles of their scores.

        Args:
            scores_all (torch.Tensor): A num_instances-length array where scores_all[i] is the score of the true class for instance i.
            labels (torch.Tensor): A num_instances-length array of true class labels.
            q (list, optional): Quantiles to include in embedding. Default is [0.5, 0.6, 0.7, 0.8, 0.9].

        Returns:
            tuple:
                - embeddings (torch.Tensor): A num_classes x len(q) array where the ith row is the embeddings of class i.
                - cts (torch.Tensor): A num_classes-length array where cts[i] is the number of times class i appears in labels.
        """
        num_classes = len(torch.unique(labels))
        embeddings = torch.zeros((num_classes, len(q)), device=self._device)
        cts = torch.zeros((num_classes,), device=self._device)

        for i in range(num_classes):
            if len(scores_all.shape) > 1:
                raise DimensionError(f"Expected 1-dimension, but got {len(scores_all.shape)}-dimension.")

            class_i_scores = scores_all[labels == i]

            cts[i] = class_i_scores.shape[0]
            for k in range(len(q)):
                # Computes the q-quantiles of samples and returns the vector of quantiles
                embeddings[i, k] = torch.kthvalue(class_i_scores, int(math.ceil(cts[i] * q[k])), dim=0).values.to(self._device)
        
        return embeddings, cts

    def __compute_cluster_specific_qhats(self, cluster_assignments, cal_class_scores, cal_true_labels, alpha):
        """
        Compute cluster-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha).

        Args:
            cluster_assignments (torch.Tensor): A num_classes-length array where entry i is the index of the cluster that class i belongs to. Rare classes can be assigned to cluster -1 and they will automatically be given as default_qhat.
            cal_class_scores (torch.Tensor): The scores for each instance in the calibration set.
            cal_true_labels (torch.Tensor): The true class labels for instances in the calibration set.
            alpha (float): Desired coverage level.

        Returns:
            torch.Tensor: A num_classes-length array where entry i is the quantile corresponding to the cluster that class i belongs to.
        """

        # Map true class labels to clusters
        cal_true_clusters = torch.tensor([cluster_assignments[label] for label in cal_true_labels], device=self._device)
        num_clusters = torch.max(cluster_assignments) + 1

        cluster_qhats = self.__compute_class_specific_qhats(cal_class_scores, cal_true_clusters, num_clusters, alpha)
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        qhats_class = torch.tensor([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)],
                                   device=self._device)

        return qhats_class

    def __compute_class_specific_qhats(self, cal_class_scores, cal_true_clusters, num_clusters, alpha):
        """
        Compute class-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha).

        Args:
            cal_class_scores (torch.Tensor): A num_instances-length array where cal_class_scores[i] is the score for instance i.
            cal_true_clusters (torch.Tensor): A num_instances-length array of true class labels. If class -1 appears, it will be assigned the null_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat.
            num_clusters (int): The number of clusters.
            alpha (float): Desired coverage level.

        Returns:
            torch.Tensor: The threshold of each class.
        """

        # Compute quantile q_hat that will result in marginal coverage of (1-alpha)
        # null_qhat = self._calculate_conformal_value(cal_class_scores, alpha)
        null_qhat = torch.inf

        q_hats = torch.zeros((num_clusters,), device=self._device)  # q_hats[i] = quantile for class i
        for k in range(num_clusters):
            # Only select data for which k is true class
            idx = (cal_true_clusters == k)
            scores = cal_class_scores[idx]
            q_hats[k] = self._calculate_conformal_value(scores, alpha)
        # print(torch.argwhere(cal_true_clusters==-1))
        if -1 in cal_true_clusters:
            q_hats = torch.concatenate((q_hats, torch.tensor([null_qhat], device=self._device)))

        return q_hats
