# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from torchcp.classification.predictors.split import SplitPredictor


class ClassWisePredictor(SplitPredictor):
    """

    Applications of Class-Conditional Conformal Predictor in Multi-Class Classification (Shi et al., 2013)
    paper: https://ieeexplore.ieee.org/document/6784618
    
        
    :param score_function: non-conformity score function.
    :param model: a pytorch model.
    """

    def __init__(self, score_function, model=None):
        super(ClassWisePredictor, self).__init__(score_function, model)
        self.q_hat = None

    def calculate_threshold(self, logits, labels, alpha):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        # Count the number of classes
        num_classes = logits.shape[1]
        self.q_hat = torch.zeros(num_classes, device=self._device)
        for label in range(num_classes):
            x_cal_tmp = logits[labels == label]
            y_cal_tmp = labels[labels == label]
            scores = self.score_function(x_cal_tmp, y_cal_tmp)
            self.q_hat[label] = self._calculate_conformal_value(scores, alpha)
