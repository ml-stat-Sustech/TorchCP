# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from torchcp.classification.predictor import SplitPredictor
from torchcp.utils.common import calculate_conformal_value


class RC3PPredictor(SplitPredictor):
    """
    Rank Calibrated Class-conditional Conformal Prediction (RC3P) as described in 
    "Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration" 
    by Shi et al., NeurIPS 2024.
    
    Args:
        score_function (callable): Non-conformity score function (e.g., APS or RAPS).
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        alpha (float, optional): The significance level. Default is 0.1.
        device (torch.device, optional): The device on which the model is located. Default is None.
    """

    def __init__(self, score_function, model=None, alpha=0.1, device=None):
        super().__init__(score_function, model, alpha=alpha, device=device)
        self.num_classes = None  # Will be set during calibration
        self.class_thresholds = None  # Store class-wise conformal thresholds Q_{1-α_y}^{class}(y)
        self.class_rank_limits = None  # Store class-wise label rank thresholds k(y)

    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha=None):
        """
        Calibrate the RC3P predictor using class-wise conformal scores and label ranks.

        Args:
            cal_dataloader (DataLoader): Calibration data loader.
            alpha (float): Target miscoverage rate (0 < alpha < 1). Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        
        # Set num_classes based on logits shape
        self.num_classes = logits.shape[1]
                
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha=None):
        """
        Perform class-wise calibration for conformal thresholds and label ranks.

        Args:
            logits (torch.Tensor): Model logits for calibration data.
            labels (torch.Tensor): True labels for calibration data.
            alpha (float): Target miscoverage rate. Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        logits = logits.to(self._device)
        labels = labels.to(self._device)
        
        num_classes = logits.shape[1]
        
        self.class_thresholds = torch.full(size=(self.num_classes,), fill_value=float('inf')).to(self._device)
        self.class_rank_limits = torch.full(size=(self.num_classes,), fill_value=self.num_classes).to(self._device)
        
        
        ranks = torch.argsort(logits, dim=1, descending=True)  # (N, K), descending order

        for y in range(num_classes):
            # Filter calibration examples for class y
            mask = (labels == y)
            class_logits = logits[mask]  # (n_y, K)
            class_labels = labels[mask]  # (n_y,)

            if class_logits.size(0) == 0:  # Skip if no examples for this class
                continue

            # Compute non-conformity scores for class y
            scores = self.score_function(class_logits, class_labels)  # (n_y,)

            # Compute class-wise top-k error (ε_y^k)
            class_ranks = ranks[mask]  # (n_y, K)
            y_tensor = torch.tensor(y, device=class_ranks.device)
            true_label_rank = (class_ranks == y_tensor.unsqueeze(-1)).nonzero(as_tuple=True)[1]  # (n_y,)
            top_k_errors = []
            for k in range(1, num_classes + 1):
                error = (true_label_rank >= k).float().mean().item()  # P(r_f(X,Y) > k | Y=y)
                top_k_errors.append(error)

            # Option II: Select minimal k(y) such that ε_y^k < alpha (Eq. 7 in paper)
            k_y = next((k + 1 for k, err in enumerate(top_k_errors) if err < alpha), num_classes)
            epsilon_y = top_k_errors[k_y - 1] if k_y <= num_classes else 0
            alpha_y = alpha - epsilon_y  # Adjusted miscoverage rate

            # Compute class-wise conformal threshold Q_{1-α_y}^{class}(y)
            q_hat_y = calculate_conformal_value(scores, alpha_y)

            # Store thresholds in lists
            self.class_thresholds[y] = q_hat_y
            self.class_rank_limits[y] = k_y

    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances using RC3P.

        Args:
            x_batch (torch.Tensor): A batch of input instances.

        Returns:
            torch.Tensor: Prediction sets for each instance in the batch (as boolean tensors).
        """
        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        x_batch = self._model(x_batch.to(self._device)).float()
        x_batch = self._logits_transformation(x_batch).detach()
        sets = self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits):
        """
        Generate prediction sets from logits using class-wise thresholds and rank limits.

        Args:
            logits (torch.Tensor): Model logits for test data (B, K).

        Returns:
            torch.Tensor: Prediction sets for each instance (as boolean tensors).
        """
        if self.class_thresholds is None:
            raise ValueError("Calibration not performed. Please run calibrate() first.")
            
        batch_size, num_classes = logits.shape
        ranks = torch.sort(logits, dim=1, descending=True)[1]  # (B, K)
        
        scores = self.score_function(logits)  # (B, K)
                
        # For each batch item and class, find where that class appears in the rank ordering
        # This gives us a batch_size x num_classes tensor of ranks (1-based)
        expanded_ranks = ranks.unsqueeze(2).expand(batch_size, num_classes, num_classes)
        
        # Expand class_indices to match the ranks dimensions for comparison
        # Shape: [batch_size, num_classes, num_classes]
        expanded_class_indices = torch.arange(num_classes, device=self._device) \
                                    .reshape(1, 1, num_classes) \
                                    .expand(batch_size, num_classes, num_classes)
        
        # Compare expanded ranks with expanded class indices
        # This gives a 3D boolean tensor where [b, p, c] is True if ranks[b, p] == c
        matches = (expanded_ranks == expanded_class_indices)
        
        # Find the positions (p) where each class (c) appears for each sample (b)
        b_indices, p_indices, c_indices = matches.nonzero(as_tuple=True)
        
        # Initialize ranks_all with zeros
        ranks_all = torch.zeros((batch_size, num_classes), dtype=torch.long, device=self._device)
        
        # Use scatter to place the position values (p_indices + 1) at the right indices in ranks_all
        # We add 1 to convert from 0-based to 1-based indexing for ranks
        ranks_all[b_indices, c_indices] = p_indices + 1
        prediction_sets = (scores <= self.class_thresholds.unsqueeze(0)) & \
                (ranks_all <= self.class_rank_limits.unsqueeze(0))
        
        return prediction_sets
