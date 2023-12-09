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

    def calibrate(self, x_cal, y_cal, alpha):
        # the number of labels
        labels_num = x_cal.shape[1]
        self.q_hats = torch.zeros(labels_num)
        for label in range(labels_num):
            scores = []
            x_cal_tmp = x_cal[y_cal == label]
            y_cal_tmp = y_cal[y_cal == label]
            for index, (x, y) in enumerate(zip(x_cal_tmp, y_cal_tmp)):
                scores.append(self.score_function(x, y))
            scores = torch.tensor(scores)
            self.q_hats[label] = torch.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])

    def predict(self, x):
        x_scores = self.score_function.predict(x)
        S = torch.argwhere(x_scores < self.q_hats).reshape(-1).tolist()
        return S
