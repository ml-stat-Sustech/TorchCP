# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from deepcp.classification.predictor.base import BasePredictor

class StandardPredictor(BasePredictor):

    def calibrate(self, model, cal_dataloader, alpha):
        logits,labels = self._cal_model_output(model, cal_dataloader)
        probs = F.softmax(logits,dim=1)
        probs = probs.numpy()
        labels = labels.numpy()
        self.calibrate_threshold(probs, labels, alpha)
        
        
    def calibrate_threshold(self, probs, labels, alpha):
        scores = np.zeros(probs.shape[0])
        for index, (x, y) in enumerate(zip(probs, labels)):
            scores[index] = self.score_function(x, y)
        self.q_hat = np.quantile(scores, np.ceil((scores.shape[0] + 1) * (1 - alpha)) / scores.shape[0])


    def predict(self, x_batch):
        logits = self.model(x_batch.to(self.model_device)).detach().cpu()
        probs_batch = F.softmax(logits,dim=1).numpy()
        sets = []
        for index, probs in enumerate(probs_batch):
            scores = self.score_function.predict(probs)
            sets.append(np.argwhere(scores < self.q_hat).reshape(-1).tolist())
        return sets
    
    def predict_with_probs(self, probs):
        scores = self.score_function.predict(probs)
        S = np.argwhere(scores < self.q_hat).reshape(-1).tolist()
        return S
    

