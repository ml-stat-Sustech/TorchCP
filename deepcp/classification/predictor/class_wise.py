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

    
    def calculate_threshold(self, logits, labels, alpha):
        # the number of labels
        labels_num = logits.shape[1]
        self.q_hat = torch.zeros(labels_num)
        for label in range(labels_num):
            x_cal_tmp = logits[labels == label]
            y_cal_tmp = labels[labels == label]
            scores = torch.zeros(x_cal_tmp.shape[0])
            for index, (x, y) in enumerate(zip(x_cal_tmp, y_cal_tmp)):
                scores[index] = self.score_function(x, y)
            
            qunatile_value = np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0]
            if qunatile_value>1:
                qunatile_value = 1
                
            self.q_hat[label] = torch.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])


