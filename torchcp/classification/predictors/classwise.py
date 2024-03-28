# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import warnings
from torchcp.classification.predictors.split import SplitPredictor


class ClassWisePredictor(SplitPredictor):
    """

    Applications of Class-Conditional Conformal Predictor in Multi-Class Classification (Shi et al., 2013)
    paper: https://ieeexplore.ieee.org/document/6784618
    
        
    :param score_function: non-conformity score function.
    :param model: a pytorch model.
    """

    def __init__(self, score_function, model=None, temperature=1):
        super(ClassWisePredictor, self).__init__(score_function, model, temperature)
        self.q_hat = None

    def calculate_threshold(self, logits, labels, alpha):
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
                

