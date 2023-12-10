# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#




import torch
import numpy as np

from deepcp.classification.predictor.base import BasePredictor


class ClassWisePredictor(BasePredictor):
    def __init__(self, score_function):
        super().__init__(score_function)

    def calibrate(self, x_cal, y_cal, alpha):
        # the number of labels
        labels_num = x_cal.shape[1]
        self.q_hats = np.zeros(labels_num)
        for label in range(labels_num):
            
            x_cal_tmp = x_cal[y_cal == label]
            y_cal_tmp = y_cal[y_cal == label]
            scores = np.zeros(x_cal_tmp.shape[0])
            for index, (x, y) in enumerate(zip(x_cal_tmp, y_cal_tmp)):
                scores[index] = self.score_function(x, y)
            self.q_hats[label] = np.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])

    def predict(self, x):
        x_scores = self.score_function.predict(x)
        S = np.argwhere(x_scores < self.q_hats).reshape(-1).tolist()
        return S
