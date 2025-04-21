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
    """
    Surrogate Conformal Predictor Optimization (SCPO).

    The class implements the loss function of the surrogate conformal predictor optimization, 
    which is an approach to train the conformal predictor directly with maximum predictive 
    efficiency as the optimization objective. The conformal predictor is approximated by a 
    differentiable objective function and gradient descent used to optimize it.

    Args:
        predictor (torchcp.classification.Predictor): An instance of the CP predictor class.
        alpha (float): The significance level for each training batch.
        lambda_val (float): Weight for the coverage loss term.
        gamma_val (float): Inverse of the temperature value.
        loss_transform (str, optional): A transform for loss. Default is "log".
            Can be "log" or "neg_inv".

    Examples::
        >>> predictor = torchcp.classification.SplitPredictor(score_function=THR(score_type="identity"))
        >>> scpo = SCPOLoss(predictor=predictor, alpha=0.01)
        >>> logits = torch.randn(100, 10)
        >>> labels = torch.randint(0, 2, (100,))
        >>> loss = scpo(logits, labels)
        >>> loss.backward()
        
    Reference:
        Bellotti et al. "Optimized conformal classification using gradient descent approximation", http://arxiv.org/abs/2105.11255
        
    """

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