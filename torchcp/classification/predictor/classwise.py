# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.predictor.split import SplitPredictor


class ClassWisePredictor(SplitPredictor):
    """
    Method: Class-wise conformal prediction
    Paper: Conditional validity of inductive conformal predictors (Vovk et al., 2012)
    Link: https://proceedings.mlr.press/v25/vovk12.html
    
        
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.

    Attributes:
        q_hat (torch.Tensor): The calibrated threshold for each class.
    """

    def __init__(self, score_function, model=None, temperature=1):

        super(ClassWisePredictor, self).__init__(score_function, model, temperature)
        self.q_hat = None

    def calculate_threshold(self, logits, labels, alpha):
        """
        Calculate the class-wise conformal prediction thresholds.

        Args:
            logits (torch.Tensor): The logits output from the model.
            labels (torch.Tensor): The ground truth labels.
            alpha (float): The significance level.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")

        alpha = torch.tensor(alpha, device=self._device)
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        # Count the number of classes
        num_classes = logits.shape[1]
        self.q_hat = torch.zeros(num_classes, device=self._device)
        scores = self.score_function(logits, labels)
        marginal_q_hat = self._calculate_conformal_value(scores, alpha)
        for label in range(num_classes):
            temp_scores = scores[labels == label]
            self.q_hat[label] = self._calculate_conformal_value(temp_scores, alpha, marginal_q_hat)
