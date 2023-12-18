# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The reference repository is https://github.com/tiffanyding/class-conditional-conformal

from collections import Counter

import torch
import numpy as np
from sklearn.cluster import KMeans

from deepcp.classification.predictor.base import BasePredictor
from deepcp.classification.predictor.base import InductivePredictor
from deepcp.utils.common import DimensionError
from deepcp.classification.utils import Metrics


class ClusterPredictor(InductivePredictor):
    """
    Clustered conformal prediction (Ding et al., 2023)
    paper: https://arxiv.org/abs/2306.09335
    """

    def __init__(self, score_function, model=None, ratio_clustering="auto", num_clusters="auto", split='random'):
        """

        :param score_function: score functions of CP
        :param model: a deep learning model
        :param cluster_ratio: The ratio of examples in the calibration dataset used to cluster classes
        :param cluster_num: The number of clusters. If cluster_ratio is "auto", the number of clusters is automatically computed.
        :param split: The method to split the dataset into clustering dataset and calibration set. split: How to split data between clustering step and calibration step. Options are 'balanced' (sample n_clustering per class), 'proportional' (sample proportional to distribution such that rarest class has n_clustering example), 'doubledip' (don't split and use all data for both steps, or 'random' (each example is assigned to clustering step with some fixed probability)
        """

        super(ClusterPredictor, self).__init__(score_function, model)
        self.__ratio_clustering = ratio_clustering
        self.__num_clusters = num_clusters
        self.__split = split

    def calculate_threshold(self, logits, labels, alpha):
        device = logits.device
        num_classes = logits.shape[1]
        scores = torch.zeros(logits.shape[0]).to(device)
        for index, (x, y) in enumerate(zip(logits, labels)):
            scores[index] = self.score_function(x, y)

        alpha = torch.tensor(alpha).to(device)
        classes_statistics = torch.tensor([torch.sum(labels == k).item() for k in range(num_classes)])

        # 1) Choose necessary parameters for Cluster algorithm
        if self.__ratio_clustering == 'auto' and self.__num_clusters == 'auto':
            n_min = torch.minimum(classes_statistics)
            n_thresh = self.__get_quantile_threshold(alpha)

            n_min = torch.maxmimum(n_min,
                                   n_thresh)  # Classes with fewer than n_thresh examples will be excluded from clustering
            num_remaining_classes = torch.sum(classes_statistics >= n_min)

            # Compute the number of clusters and the minium number of examples for each class
            n_clustering = int(n_min * num_remaining_classes / (75 + num_remaining_classes))
            self.__num_clusters = int(torch.floor(n_clustering / 2))
            self.__ratio_clustering = n_clustering / n_min

        # 2) Split data
        if self.__split == 'proportional':
            # Split dataset along with fraction "frac_clustering"
            n_k = torch.tensor([self.__ratio_clustering * classes_statistics[k] for k in range(num_classes)])
            clustering_scores, clustering_labels, cal_scores, cal_labels = self.__split_X_and_y(scores,
                                                                                                labels,
                                                                                                n_k,
                                                                                                num_classes=num_classes)

        elif self.__split == 'doubledip':
            clustering_scores, clustering_labels = scores, labels
            cal_scores, cal_labels = scores, labels
        elif self.__split == 'random':
            # Each point is assigned to clustering set w.p. frac_clustering 
            idx1 = torch.rand(size=(len(labels),)) < self.__ratio_clustering
            clustering_scores = scores[idx1]
            clustering_labels = labels[idx1]
            cal_scores = scores[~idx1]
            cal_labels = labels[~idx1]

        else:
            raise Exception('Invalid split. Options are proportional, doubledip, and random')

        # 3)  Identify "rare" classes
        rare_classes = self.__get_rare_classes(clustering_labels, alpha, num_classes)

        # 4) Run clustering
        if num_classes - len(rare_classes) > self.__num_clusters and self.__num_clusters > 1:
            # Filter out rare classes and re-index
            remaining_idx, filtered_labels, class_remapping = self.__remap_classes(clustering_labels, rare_classes)
            filtered_scores = clustering_scores[remaining_idx]

            # Compute embedding for each class and get class counts
            embeddings, class_cts = self.__embed_all_classes(filtered_scores, filtered_labels,
                                                             q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True)
            kmeans = KMeans(n_clusters=int(self.__num_clusters), n_init=10).fit(embeddings,
                                                                                sample_weight=np.sqrt(class_cts))
            nonrare_class_cluster_assignments = kmeans.labels_

            cluster_assignments = -np.ones((num_classes,), dtype=int)
            for cls, remapped_cls in class_remapping.items():
                cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
        else:
            cluster_assignments = -np.ones((num_classes,), dtype=int)

        # 5) Compute qhats for each cluster

        self.q_hat = self.__compute_cluster_specific_qhats(cluster_assignments,
                                                           cal_scores,
                                                           cal_labels,
                                                           alpha=alpha)

    def __get_quantile_threshold(self, alpha):
        '''
        Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
        '''
        device = alpha.device
        n = torch.tensor(0).to(device)
        while torch.ceil((n + 1) * (1 - alpha) / n) > 1:
            n += 1
        return n

    def __split_X_and_y(self, X, y, n_k, num_classes):

        if len(X.shape) == 2:
            X1 = torch.zeros((torch.sum(n_k), X.shape[1]))
        else:
            X1 = torch.zeros((torch.sum(n_k),))
        y1 = torch.zeros((torch.sum(n_k),), dtype=torch.int32)

        all_selected_indices = torch.zeros(y.shape)

        i = 0
        for k in range(num_classes):
            # Randomly select n instances of class k
            idx = torch.argwhere(y == k).flatten()
            #         pdb.set_trace()
            selected_idx = torch.random.choice(idx, replace=False, size=(n_k[k],))

            X1[i:i + n_k[k]] = X[selected_idx]
            y1[i:i + n_k[k]] = k
            i += n_k[k]

            all_selected_indices[selected_idx] = 1

        X2 = X[all_selected_indices == 0]
        y2 = y[all_selected_indices == 0]

        return X1, y1, X2, y2

    def __get_rare_classes(self, labels, alpha, num_classes):
        """
        Choose classes whose number is less than or equal to threshold.

        """
        thresh = self.__get_quantile_threshold(alpha)
        classes, _, cts = torch.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh]

        # Also included any classes that are so rare that we have 0 labels for it

        all_classes = torch.arange(num_classes)
        zero_ct_classes = all_classes[(all_classes.view(1, -1) != classes.view(-1, 1)).all(dim=0)]
        rare_classes = torch.concatenate((rare_classes, zero_ct_classes))

        return rare_classes

    def __remap_classes(self, labels, rare_classes):
        '''
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed

        Outputs:
            - remaining_idx: Boolean array the same length as labels. Entry i is True
            iff labels[i] is not in rare_classes 
            - remapped_labels: Array that only contains the entries of labels that are 
            not in rare_classes (in order) 
            - remapping: Dict mapping old class index to new class index

        '''
        remaining_idx = ~torch.isin(labels, rare_classes)

        remaining_labels = labels[remaining_idx]
        remapped_labels = torch.zeros(remaining_labels.shape, dtype=torch.int32)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1
        return remaining_idx, remapped_labels, remapping

    def __embed_all_classes(self, scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
        '''
        Input:
            - scores_all: num_instances x num_classes array where 
                scores_all[i,j] = score of class j for instance i
            Alternatively, num_instances-length array where scores_all[i] = score of true class for instance i
            - labels: num_instances-length array of true class labels
            - q: quantiles to include in embedding
            - return_cts: if True, return an array containing the counts for each class 
            
        Output: 
            - embeddings: num_classes x len(q) array where ith row is the embeddings of class i
            - (Optional) cts: num_classes-length array where cts[i] = # of times class i 
            appears in labels 
        '''

        def quantile_embedding(samples, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
            '''
            Computes the q-quantiles of samples and returns the vector of quantiles
            '''
            return np.quantile(samples, q)

        num_classes = len(np.unique(labels))

        embeddings = np.zeros((num_classes, len(q)))
        cts = np.zeros((num_classes,))

        for i in range(num_classes):
            if len(scores_all.shape) > 1:
                raise DimensionError(f"Expected 1-dimension, but got {len(scores_all.shape)}-dimension.")

            class_i_scores = scores_all[labels == i]

            cts[i] = class_i_scores.shape[0]
            embeddings[i, :] = quantile_embedding(class_i_scores, q=q)

        if return_cts:
            return embeddings, cts
        else:
            return embeddings

    def __compute_cluster_specific_qhats(self, cluster_assignments, cal_class_scores, cal_true_labels, alpha):
        '''
        Computes cluster-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha)
        
        Inputs:
            - cluster_assignments: num_classes length array where entry i is the index of the cluster that class i belongs to.
            Clusters should be 0-indexed. Rare classes can be assigned to cluster -1 and they will automatically be given
            qhat_k = default_qhat. 
            - cal_class_scores: num_instances x num_classes array where class_scores[i,j] = score of class j for instance i.
            Alternatively, a num_instances-length array of conformal scores for true class
            - cal_true_labels: num_instances length array of true class labels (0-indexed)
            - alpha: Determines desired coverage level
            - null_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to null_qhat.
            If null_qhat == 'standard', we compute the qhat for standard conformal and use that as the default value
        Output:
            num_classes length array where entry i is the quantile correspond to the cluster that class i belongs to. 
            All classes in the same cluster have the same quantile.
        '''
        # If we want the null_qhat to be the standard qhat, we should compute this before we remap the values
        null_qhat = self.__compute_qhat(cal_class_scores, cal_true_labels, alpha)


        # Edge case: all cluster_assignments are -1. 
        if np.all(cluster_assignments == -1):
            return null_qhat * torch.ones(cluster_assignments.shape)

        # Map true class labels to clusters
        cal_true_clusters = torch.Tensor([cluster_assignments[label] for label in cal_true_labels])

        # Compute cluster qhats

        cluster_qhats = self.__compute_class_specific_qhats(
            cal_class_scores,
            cal_true_clusters,
            alpha=alpha,
            num_classes=np.max(cluster_assignments) + 1,
            null_qhat=null_qhat,
            default_qhat=np.inf,

        )
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        class_qhats = torch.Tensor([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])

        return class_qhats

    def __compute_qhat(self, class_scores, true_labels, alpha):
        '''
        Compute quantile q_hat that will result in marginal coverage of (1-alpha)
        
        Inputs:
            class_scores:  num_instances-length array of  conformal scores for true class. A higher score indicates more uncertainty
            true_labels: num_instances length array of ground truth labels
        
        '''
        # Select scores that correspond to correct label

        # Sort scores
        scores = np.sort(class_scores)

        # Identify score q_hat such that ~(1-alpha) fraction of scores are below qhat 
        #    Note: More precisely, it is (1-alpha) times a small correction factor
        n = len(true_labels)
        q_hat = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method='inverted_cdf')

        return q_hat

    def __compute_class_specific_qhats(self, cal_class_scores, cal_true_labels, num_classes, alpha, null_qhat,
                                       default_qhat):
        '''
        Computes class-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha)
        
        Inputs:
            - cal_class_scores: 
                num_instances x num_classes array where cal_class_scores[i,j] = score of class j for instance i
                OR
                num_instances length array where entry i is the score of the true label for instance i
            - cal_true_labels: num_instances-length array of true class labels (0-indexed). If class -1 appears,
            it will be assigned the default_qhat value. It is appended as an extra entry of the returned q_hats
            so that q_hats[-1] = null_qhat
            - alpha: Determines desired coverage level
            - default_qhat: Float, or 'standard'. For classes that do not appear in cal_true_labels, the class 
            specific qhat is set to default_qhat. If default_qhat == 'standard', we compute the qhat for standard
            conformal and use that as the default value
            - null_qhat: Float, or 'standard', as described for default_qhat. Only used if -1 appears in
            cal_true_labels. null_qhat is assigned to 
            class/cluster -1 
        '''

        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats 
        # to get a marginal coverage guarantee

        num_samples = len(cal_true_labels)
        q_hats = np.zeros((num_classes,))  # q_hats[i] = quantile for class i
        class_cts = np.zeros((num_classes,))
        for k in range(num_classes):

            # Only select data for which k is true class
            idx = (cal_true_labels == k)

            if len(cal_class_scores.shape) == 2:
                scores = cal_class_scores[idx, k]
            else:
                scores = cal_class_scores[idx]

            class_cts[k] = scores.shape[0]

            if len(scores) == 0:
                assert default_qhat is not None, f"Class/cluster {k} does not appear in the calibration set, so the quantile for this class cannot be computed. Please specify a value for default_qhat to use in this case."
                # print(f'Warning: Class/cluster {k} does not appear in the calibration set,', 
                # f'so default q_hat value of {default_qhat} will be used')
                q_hats[k] = default_qhat
            else:
                scores = np.sort(scores)
                num_samples = len(scores)
                val = np.ceil((num_samples + 1) * (1 - alpha)) / num_samples
                if val > 1:
                    assert default_qhat is not None, f"Class/cluster {k} does not appear enough times to compute a proper quantile. Please specify a value for default_qhat to use in this case."
                    # print(f'Warning: Class/cluster {k} does not appear enough times to compute a proper quantile,', 
                    # f'so default q_hat value of {default_qhat} will be used')
                    q_hats[k] = default_qhat
                else:
                    q_hats[k] = np.quantile(scores, val, method='inverted_cdf')

        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))

        return q_hats
