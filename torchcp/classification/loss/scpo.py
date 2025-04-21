# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__all__ = ["SCPO"]

import torch
import torch.nn.functional as F

from torchcp.classification.loss.confts import ConfTSLoss
from torchcp.classification.loss.conftr import ConfTrLoss


class SCPOLoss(ConfTSLoss):

    def __init__(self, predictor, alpha, lambda_val=500, gamma_val=5):
        super(SCPOLoss, self).__init__(predictor, alpha)
        self.lambda_val = lambda_val
        self.gamma_val = gamma_val

        self.size_loss_fn = ConfTrLoss(predictor, 
                                       alpha, 
                                       fraction=0.5, 
                                       epsilon=1/gamma_val, 
                                       loss_type="valid", 
                                       target_size=0, 
                                       loss_transform="abs")
        self.coverage_loss_fn = ConfTrLoss(predictor, 
                                           alpha, 
                                           fraction=0.5, 
                                           epsilon=1/gamma_val, 
                                           loss_type="coverage")

    def forward(self, logits, labels):
        train_scores = self.predictor.score_function(logits.to(self.device))
        train_labels = labels.to(self.device)

        size_loss = self.size_loss_fn.compute_loss(train_scores, train_labels, 1)
        coverage_loss = self.coverage_loss_fn.compute_loss(train_scores, train_labels, 1)
        return torch.log(size_loss + self.lambda_val * coverage_loss)