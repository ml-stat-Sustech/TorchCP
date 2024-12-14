# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CDLoss"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .base import BaseLoss


class CDLoss(BaseLoss):
    """
    Implementation of Conformal Discriminative Loss (CDLoss) for efficient conformal prediction.
    
    This loss function encourages the model to output prediction sets that:
    1. Contain the true label with high probability
    2. Are as small as possible for efficiency
    
    The loss is computed by measuring the probability of each class being included
    in the prediction set relative to the true label's score.

    Args:
        predictor (torchcp.classification.Predictor): Predictor instance that defines
            the scoring mechanism for conformal prediction.
        epsilon (float, optional): Temperature parameter that controls the sharpness
            of the sigmoid function. Smaller values create sharper boundaries. 
            Default: 1e-4

    Reference:
        Liu et al. "C-Adapter: Adapting Deep Classifiers for Efficient Conformal 
        Prediction Sets". arXiv:2410.09408, 2024.
        
    """

    def __init__(self, predictor, epsilon=1e-4):
        super(CDLoss, self).__init__(predictor)
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")

        self.epsilon = epsilon
        self.predictor = predictor

    def forward(self, logits, labels):
        """
        Compute the Conformal Discriminative Loss for a batch of predictions.
        
        Args:
            logits (Tensor): Model output logits with shape (batch_size, num_classes)
            labels (Tensor): Ground truth class labels with shape (batch_size,)
        
        Returns:
            Tensor: Scalar loss value computed as the weighted average of prediction
                set probabilities across all classes and samples.
        
        Note:
            Implementation follows Equation (4) from the paper, using sigmoid function
            to compute smooth approximation of prediction set membership.
        """
        all_scores = self.predictor.score_function(logits)
        label_scores = self.predictor.score_function(logits, labels)
        label_scores = label_scores.unsqueeze(1).expand_as(all_scores)
        # Computing the probability of each label contained in the prediction set.
        pred_sets = torch.sigmoid((all_scores - label_scores) / self.epsilon)
        loss = torch.mean(pred_sets)

        return loss
