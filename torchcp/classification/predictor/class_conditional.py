# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.predictor.split import SplitPredictor


class ClassConditionalPredictor(SplitPredictor):
    """
    Method: Class-conditional conformal prediction
    Paper: Conditional validity of inductive conformal predictors (Vovk et al., 2012)
    Link: https://proceedings.mlr.press/v25/vovk12.html
    
        
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        alpha (float, optional): The significance level. Default is 0.1.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
        device (torch.device, optional): The device on which the model is located. Default is None.

    Attributes:
        q_hat (torch.Tensor): The calibrated threshold for each class.
    """

    def __init__(self, score_function, model=None, alpha=0.1, temperature=1, device=None):

        super(ClassConditionalPredictor, self).__init__(score_function, model, alpha, temperature, device)
        self.q_hat = None

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

        alpha = torch.tensor(alpha, device=self._device)
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        # Count the number of classes
        num_classes = logits.shape[1]
        self.q_hat = torch.zeros(num_classes, device=self._device)
        scores = self.score_function(logits, labels)
        for label in range(num_classes):
            temp_scores = scores[labels == label]
            self.q_hat[label] = self._calculate_conformal_value(temp_scores, alpha)
