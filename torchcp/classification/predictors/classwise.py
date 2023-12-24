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
    """

    def __init__(self, score_function, model=None):
        super(ClassWisePredictor, self).__init__(score_function, model)
        self.q_hat = None

    def calculate_threshold(self, logits, labels, alpha):
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        # Count the number of classes
        labels_num = logits.shape[1]
        self.q_hat = torch.zeros(labels_num, device=self._device)
        for label in range(labels_num):
            x_cal_tmp = logits[labels == label]
            y_cal_tmp = labels[labels == label]
            scores = logits.new_zeros(x_cal_tmp.shape[0])
            for index, (x, y) in enumerate(zip(x_cal_tmp, y_cal_tmp)):
                scores[index] = self.score_function(x, y)

            self.q_hat[label] = self._calculate_conformal_value(scores, alpha)
