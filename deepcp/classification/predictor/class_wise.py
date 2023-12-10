# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#




import torch
import torch.nn.functional as F
import numpy as np

from deepcp.classification.predictor.standard import StandardPredictor


class ClassWisePredictor(StandardPredictor):
    def __init__(self, score_function):
        super().__init__(score_function)
    
    def calibrate_threshold(self, probs, labels, alpha):
        # the number of labels
        labels_num = probs.shape[1]
        self.q_hat = np.zeros(labels_num)
        for label in range(labels_num):
            x_cal_tmp = probs[labels == label]
            y_cal_tmp = labels[labels == label]
            scores = np.zeros(x_cal_tmp.shape[0])
            for index, (x, y) in enumerate(zip(x_cal_tmp, y_cal_tmp)):
                scores[index] = self.score_function(x, y)
            self.q_hat[label] = np.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])


