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
from deepcp.classification.predictor.class_wise import StandardPredictor
from deepcp.utils.common import DimensionError
from deepcp.classification.utils import Metrics


class ClusterPredictor(StandardPredictor):
    def __init__(self, score_function, model, seed, cluster_ratio= "auto", cluster_num = "auto", split= 'random'):
        """_summary_

        Args:
            score_function (_type_): _description_
            seed (_type_): random seed
            cluster_ratio (float, optional): the ratio of examples for clustering step. Defaults to 0.2.
            split (str, optional):split: How to split data between clustering step and calibration step. Options are 'balanced' (sample n_clustering per class), 'proportional' (sample proportional to distribution such that rarest class has n_clustering example), 'doubledip' (don't split and use all data for both steps, or 'random' (each example is assigned to clustering step with some fixed probability) 
        """
        super(ClusterPredictor, self).__init__(score_function, model)
        self.__seed = seed
        self.__cluster_ratio =  cluster_ratio
        self.__cluster_num =  cluster_num
        self.__split =  split

        
    def calculate_threshold(self, probs, labels, alpha):

        num_classes = probs.shape[1]
        scores = np.zeros(probs.shape[0])
        for index, (x, y) in enumerate(zip(probs, labels)):
            scores[index] = self.score_function(x, y)

        
        if self.__cluster_ratio == 'auto' and self.__cluster_num == 'auto':
            list_y_cal = labels.tolist()

            cts_dict = Counter(list_y_cal)
            cts = [cts_dict.get(k, 0) for k in range(num_classes)]
            n_min = min(cts)
            n_thresh = self.__get_quantile_threshold(alpha) 

            
            n_min = max(n_min, n_thresh) # Classes with fewer than n_thresh examples will be excluded from clustering
            num_remaining_classes = np.sum(np.array(list(cts)) >= n_min)

            # Compute the number of clusters and the minium number of examples for each class
            n_clustering, num_clusters = self.__get_clustering_parameters(num_remaining_classes, n_min)
            print(f'n_clustering={n_clustering}, num_clusters={num_clusters}')
            # Convert n_clustering to fraction relative to n_min
            frac_clustering = n_clustering / n_min

        
        # 2a) Split data
        if self.__split == 'proportional':
            n_k = [int(frac_clustering*cts[k]) for k in range(num_classes)]
            clustering_scores, clustering_labels, cal_scores, cal_labels = self.__split_X_and_y(scores, 
                                                            labels, 
                                                            n_k, 
                                                            num_classes=num_classes, 
                                                            seed=self.__seed)
    #                                                            split=split, # Balanced or stratified sampling 
        elif self.__split == 'doubledip':
            clustering_scores, clustering_labels = scores, labels
            cal_scores, cal_labels = scores, labels
        elif self.__split == 'random':
            # Each point is assigned to clustering set w.p. frac_clustering 
            idx1 = np.random.uniform(size=(len(labels),)) < frac_clustering 
            clustering_scores = scores[idx1]
            clustering_labels = labels[idx1]
            cal_scores = scores[~idx1]
            cal_labels = labels[~idx1]
            
        else:
            raise Exception('Invalid split. Options are balanced, proportional, doubledip, and random')

        
        # 2b)  Identify "rare" classes = classes that have fewer than 1/alpha - 1 examples 
        # in the clustering set 
        rare_classes = self.__get_rare_classes(clustering_labels, alpha, num_classes)
        print(f'{len(rare_classes)} of {num_classes} classes are rare in the clustering set'
            ' and will be assigned to the null cluster')
        
        # 3) Run clustering
        if num_classes - len(rare_classes) > num_clusters and num_clusters > 1:  
            # Filter out rare classes and re-index
            remaining_idx, filtered_labels, class_remapping = self.__remap_classes(clustering_labels, rare_classes)
            filtered_scores = clustering_scores[remaining_idx]
            
            # Compute embedding for each class and get class counts
            embeddings, class_cts = self.__embed_all_classes(filtered_scores, filtered_labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=True)
        
            kmeans = KMeans(n_clusters=int(num_clusters), random_state=self.__seed, n_init=10).fit(embeddings, sample_weight=np.sqrt(class_cts))
            nonrare_class_cluster_assignments = kmeans.labels_  

            # breakpoint()
            # Print cluster sizes
            print(f'Cluster sizes:', [x[1] for x in Counter(nonrare_class_cluster_assignments).most_common()])

            # Remap cluster assignments to original classes. Any class not included in kmeans clustering is a rare 
            # class, so we will assign it to cluster "-1" = num_clusters by Python indexing
            cluster_assignments = -np.ones((num_classes,), dtype=int)
            for cls, remapped_cls in class_remapping.items():
                cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
        else: 
            cluster_assignments = -np.ones((num_classes,), dtype=int)
            print('Skipped clustering because the number of clusters requested was <= 1')
            
        # 4) Compute qhats for each cluster

       
        self.q_hat = self.__compute_cluster_specific_qhats(cluster_assignments, 
                cal_scores, 
                cal_labels, 
                alpha= alpha, 
                null_qhat = 'standard')


    
    def __get_quantile_threshold(self,alpha):
        '''
        Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1
        '''
        n = 1
        while np.ceil((n+1)*(1-alpha)/n) > 1:
            n += 1
        return n
    
    def __get_clustering_parameters(self,num_classes, n_totalcal):
        '''
        Returns a guess of good values for num_clusters and n_clustering based solely 
        on the number of classes and the number of examples per class. 
        
        This relies on two heuristics:
        1) We want at least 150 points per cluster on average
        2) We need more samples as we try to distinguish between more distributions. 
        To distinguish between 2 distribution, want at least 4 samples per class. 
        To distinguish between 5 distributions, want at least 10 samples per class. 
        
        Output: n_clustering, num_clusters
        
        '''
        # Alias for convenience
        K = num_classes
        N = n_totalcal
        
        n_clustering = int(N*K/(75+K))
        num_clusters = int(np.floor(n_clustering / 2))
        
        return n_clustering, num_clusters
    
    
    
    # Used for creating balanced or stratified calibration dataset
    def __split_X_and_y(self,X, y, n_k, num_classes, seed=0, split='balanced'):
        '''
        Randomly generate two subsets of features X and corresponding labels y such that the
        first subset contains n_k instances of each class k and the second subset contains all
        other instances 
        
        Inputs:
            X: n x d array (e.g., matrix of softmax vectors)
            y: n x 1 array
            n_k: positive int or n x 1 array
            num_classes: total number of classes, corresponding to max(y)
            seed: random seed
            
        Output:
            X1, y1
            X2, y2
        '''
        np.random.seed(self.__seed)
        
        if split == 'balanced':
        
            if not hasattr(n_k, '__iter__'):
                n_k = n_k * np.ones((num_classes,), dtype=int)
        elif split == 'proportional':
            assert not hasattr(n_k, '__iter__')
            
            # Compute what fraction of the rarest class n_clustering corresponds to,
            # then set n_k = frac * (total # of cal points for class k)
            cts = Counter(y)
            rarest_class_ct = cts.most_common()[-1][1]
            frac = n_k / rarest_class_ct
            n_k = [int(frac*cts[k]) for k in range(num_classes)]
            
        else: 
            raise Exception('Valid split options are "balanced" or "proportional"')
                
        
        if len(X.shape) == 2:
            X1 = np.zeros((np.sum(n_k), X.shape[1]))
        else:
            X1 = np.zeros((np.sum(n_k),))
        y1 = np.zeros((np.sum(n_k), ), dtype=np.int32)
        
        all_selected_indices = np.zeros(y.shape)

        i = 0
        for k in range(num_classes):

            # Randomly select n instances of class k
            idx = np.argwhere(y==k).flatten()
    #         pdb.set_trace()
            selected_idx = np.random.choice(idx, replace=False, size=(n_k[k],))

            X1[i:i+n_k[k]] = X[selected_idx]
            y1[i:i+n_k[k]] = k
            i += n_k[k]
            
            all_selected_indices[selected_idx] = 1
            
        X2 = X[all_selected_indices == 0]
        y2 = y[all_selected_indices == 0]
        
        return X1, y1, X2, y2
    
    def __get_rare_classes(self,labels, alpha, num_classes):
        thresh = self.__get_quantile_threshold(alpha)
        classes, cts = np.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh]
        
        # Also included any classes that are so rare that we have 0 labels for it
        zero_ct_classes = np.setdiff1d(np.arange(num_classes), classes)
        rare_classes = np.concatenate((rare_classes, zero_ct_classes))
        
        return rare_classes
    
    
    def __remap_classes(self,labels, rare_classes):
        '''
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed

        Outputs:
            - remaining_idx: Boolean array the same length as labels. Entry i is True
            iff labels[i] is not in rare_classes 
            - remapped_labels: Array that only contains the entries of labels that are 
            not in rare_classes (in order) 
            - remapping: Dict mapping old class index to new class index

        '''
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
        return remaining_idx, remapped_labels, remapping
    
    
    def __embed_all_classes(self,scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9], return_cts=False):
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

            class_i_scores = scores_all[labels==i] 
                
            cts[i] = class_i_scores.shape[0]
            embeddings[i,:] = quantile_embedding(class_i_scores, q=q)
        
        if return_cts:
            return embeddings, cts
        else:
            return embeddings


    def __compute_cluster_specific_qhats(self, cluster_assignments, cal_class_scores, cal_true_labels, alpha, 
                                   null_qhat ='standard'):
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
        if null_qhat == 'standard':
            null_qhat = self.__compute_qhat(cal_class_scores, cal_true_labels, alpha)
                
        # Extract conformal scores for true labels if not already done
        if len(cal_class_scores) == 2:
            cal_class_scores = cal_class_scores[np.arange(len(cal_true_labels)), cal_true_labels]
            
        # Edge case: all cluster_assignments are -1. 
        if np.all(cluster_assignments==-1):
            
            return null_qhat * np.ones(cluster_assignments.shape)
        
        # Map true class labels to clusters
        cal_true_clusters = np.array([cluster_assignments[label] for label in cal_true_labels])
        
        # Compute cluster qhats
        
        cluster_qhats = self.__compute_class_specific_qhats(cal_class_scores, cal_true_clusters, 
                                                    alpha=alpha, num_classes=np.max(cluster_assignments)+1,
                                                    default_qhat=np.inf,
                                                    null_qhat=null_qhat)                            
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        class_qhats = np.array([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)])

        return class_qhats
        
    def __compute_qhat(self,class_scores, true_labels, alpha, plot_scores=False):
        '''
        Compute quantile q_hat that will result in marginal coverage of (1-alpha)
        
        Inputs:
            class_scores: num_instances x num_classes array of scores, or num_instances-length array of 
            conformal scores for true class. A higher score indicates more uncertainty
            true_labels: num_instances length array of ground truth labels
        
        '''
        # Select scores that correspond to correct label
        if len(class_scores.shape) == 2:
            scores = np.squeeze(np.take_along_axis(class_scores, np.expand_dims(true_labels, axis=1), axis=1))
        else:
            scores = class_scores
        
        # Sort scores
        scores = np.sort(scores)

        # Identify score q_hat such that ~(1-alpha) fraction of scores are below qhat 
        #    Note: More precisely, it is (1-alpha) times a small correction factor
        n = len(true_labels)
        q_hat = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n, method='inverted_cdf')
        


        return q_hat
    
    def __compute_class_specific_qhats(self,cal_class_scores, cal_true_labels, num_classes, alpha, 
                                 default_qhat=np.inf, null_qhat='standard', regularize=False):
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
            - regularize: If True, shrink the class-specific qhat towards the default_qhat value. Amount of
            shrinkage for a class is determined by number of samples of that class
        '''
        
        if default_qhat == 'standard':
            default_qhat = self.__compute_qhat(cal_class_scores, cal_true_labels, alpha=alpha)
            
        if null_qhat == 'standard':
            null_qhat = self.__compute_qhat(cal_class_scores, cal_true_labels, alpha=alpha)
        
        # If we are regularizing the qhats, we should set aside some data to correct the regularized qhats 
        # to get a marginal coverage guarantee
        if regularize:
            num_reserve = min(1000, int(.5*len(cal_true_labels))) # Reserve 1000 or half of calibration set, whichever is smaller
            idx = np.random.choice(np.arange(len(cal_true_labels)), size=num_reserve, replace=False)
            idx2 = ~np.isin(np.arange(len(cal_true_labels)), idx)
            reserved_scores, reserved_labels = cal_class_scores[idx], cal_true_labels[idx]
            cal_class_scores, cal_true_labels = cal_class_scores[idx2], cal_true_labels[idx2]
                    
        num_samples = len(cal_true_labels)
        q_hats = np.zeros((num_classes,)) # q_hats[i] = quantile for class i
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
                print(f'Warning: Class/cluster {k} does not appear in the calibration set,', 
                    f'so default q_hat value of {default_qhat} will be used')
                q_hats[k] = default_qhat
            else:
                scores = np.sort(scores)
                num_samples = len(scores)
                val = np.ceil((num_samples+1)*(1-alpha)) / num_samples
                if val > 1:
                    assert default_qhat is not None, f"Class/cluster {k} does not appear enough times to compute a proper quantile. Please specify a value for default_qhat to use in this case."
                    print(f'Warning: Class/cluster {k} does not appear enough times to compute a proper quantile,', 
                        f'so default q_hat value of {default_qhat} will be used')
                    q_hats[k] = default_qhat
                else:
                    q_hats[k] = np.quantile(scores, val, method='inverted_cdf')
                    
        if -1 in cal_true_labels:
            q_hats = np.concatenate((q_hats, [null_qhat]))
        
        # Optionally apply shrinkage 
        if regularize:
            N = num_classes
            n_k = np.maximum(class_cts, 1) # So that classes that never appear do not cause division by 0 issues. 
            shrinkage_factor = .03 * n_k # smaller = less shrinkage
            shrinkage_factor = np.minimum(shrinkage_factor, 1)
            print('SHRINKAGE FACTOR:', shrinkage_factor)  
            print(np.min(shrinkage_factor), np.max(shrinkage_factor))
            q_hats = default_qhat + shrinkage_factor * (q_hats - default_qhat)
            
            # Correct qhats via additive factor to achieve marginal coverage
            q_hats = self.__reconformalize(q_hats, reserved_scores, reserved_labels, alpha)

        
        return q_hats
    
    # Additive version
    def __reconformalize(self,qhats, scores, labels, alpha, adjustment_min=-1, adjustment_max=1):
        '''
        Adjust qhats by additive factor so that marginal coverage of 1-alpha is achieved
        '''
        print('Applying additive adjustment to qhats')
        # ===== Perform binary search =====
        # Convergence criteria: Either (1) marginal coverage is within tol of desired or (2)
        # quantile_min and quantile_max differ by less than .001, so there is no need to try 
        # to get a more precise estimate
        tol = 0.0005

        marginal_coverage = 0
        while np.abs(marginal_coverage - (1-alpha)) > tol:

            adjustment_guess = (adjustment_min +  adjustment_max) / 2
            print(f"\nCurrent adjustment: {adjustment_guess:.6f}")

            curr_qhats = qhats + adjustment_guess 

            preds = self._generate_prediction_set(scores, curr_qhats)
            metrics = Metrics()
            marginal_coverage = metrics('coverage_rate')(labels, preds)
            print(f"Marginal coverage: {marginal_coverage:.4f}")

            if marginal_coverage > 1 - alpha:
                adjustment_max = adjustment_guess
            else:
                adjustment_min = adjustment_guess
            print(f"Search range: [{adjustment_min}, {adjustment_max}]")

            if adjustment_max - adjustment_min < .00001:
                adjustment_guess = adjustment_max # Conservative estimate, which ensures coverage
                print("Adequate precision reached; stopping early.")
                break
                
        print('Final adjustment:', adjustment_guess)
        qhats += adjustment_guess
        
        return qhats
    