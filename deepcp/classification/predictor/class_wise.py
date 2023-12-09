import torch
import numpy as np

from deepcp.classification.predictor.base import BasePredictor


class ClassWisePredictor(BasePredictor):
    def __init__(self, score_function):
        super().__init__(score_function)


    def fit(self, x_cal, y_cal, alpha):
        # the number of labels
        labels_num = x_cal.shape[1]
        self.q_hats = torch.zeros(labels_num)
        for index,label in enumerate(self.q_hats):
            scores = []
            x_cal_tmp = x_cal[y_cal==label]
            y_cal_tmp = y_cal[y_cal==label]
            for index, (x, y) in enumerate(zip(x_cal_tmp, y_cal_tmp)):
                scores.append(self.score_function(x, y))
            scores = torch.tensor(scores)

            self.q_hats[index] = torch.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])

    def predict(self, x):
        x_scores = self.score_function.predict(x)
        S = torch.argwhere(x_scores < self.q_hats).reshape(-1).tolist()
        return S
