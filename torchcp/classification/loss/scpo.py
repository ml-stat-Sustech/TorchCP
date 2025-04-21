# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__all__ = ["SCPO"]

import torch

from torchcp.classification.loss.confts import ConfTSLoss
from torchcp.classification.loss.conftr import ConfTrLoss


class SCPOLoss(ConfTSLoss):

    def __init__(self, predictor, alpha, lambda_val=500, gamma_val=5, loss_transform="log"):
        super(SCPOLoss, self).__init__(predictor, alpha)
        self.lambda_val = lambda_val

        if loss_transform == "log":
            self.transform = torch.log
        elif loss_transform == "neg_inv":
            self.transform = lambda x: -1 / x
        else:
            raise ValueError("loss_transform should be log or neg_inv.")

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
        logits = logits.to(self.device)
        labels = labels.to(self.device)

        test_scores = self.predictor.score_function(logits)
        test_labels = labels

        return self.compute_loss(test_scores, test_labels, 1)
    
    def compute_loss(self, test_scores, test_labels, tau):
        size_loss = self.size_loss_fn.compute_loss(test_scores, test_labels, tau)
        coverage_loss = self.coverage_loss_fn.compute_loss(test_scores, test_labels, tau)
        return self.transform(size_loss + self.lambda_val * coverage_loss)