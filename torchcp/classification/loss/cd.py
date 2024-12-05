import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
__all__ = ["CDLoss"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .base import BaseLoss

class CDLoss(BaseLoss):
    """
    Method: Conformal Discriminative Loss  (CDLoss)
    Paper: C-Adapter: Adapting Deep Classifiers for Efficient Conformal Prediction Sets (Liu et al., 2024)
    Link: https://arxiv.org/abs/2410.09408
    

    Args:
        weight (float): The weight of each loss function. Must be greater than 0.
        predictor (torchcp.classification.Predictor): An instance of the CP predictor class.
        epsilon (float, optional): A temperature value. Default is 1e-4.
    """

    def __init__(self, weight, predictor, epsilon = 1e-4):

        super(CDLoss, self).__init__(weight, predictor)
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")
        
        self.epsilon  = epsilon
        self.weight = weight
        self.predictor = predictor
        
    def forward(self, logits, labels):

        all_scores = self.predictor.score_function(logits)
        label_scores = self.predictor.score_function(logits, labels)
        label_scores = label_scores.unsqueeze(1).expand_as(all_scores)
        # Computing the probability of each label contained in the prediction set.
        pred_sets = torch.sigmoid((all_scores-label_scores)/self.epsilon)
        loss = self.weight * torch.mean(pred_sets)
        
        return loss

