# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import numpy as np

from deepcp.classification.predictor.base import BasePredictor


class StandardPredictor(BasePredictor):

    def calibrate(self, x_cal, y_cal, alpha):
        scores = []
        for index, (x, y) in enumerate(zip(x_cal, y_cal)):
            scores.append(self.score_function(x, y))
        scores = torch.tensor(scores)

        self.q_hat = torch.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])
        print(self.q_hat)

    def predict(self, x):
        scores = self.score_function.predict(x)
        S = torch.argwhere(scores < self.q_hat).reshape(-1).tolist()
        return S
