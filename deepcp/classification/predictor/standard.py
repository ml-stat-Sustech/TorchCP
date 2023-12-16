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
from deepcp.classification.utils.conf_calibration import TS


class StandardPredictor(BasePredictor):
    def __init__(self, score_function, model= None, temperature = 1):
        super().__init__(score_function, model)

    
    #############################
    # The confidence calibration process
    ############################
        
        
    
    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for  examples in tqdm(cal_dataloader):
                tmp_x, tmp_labels = examples[0].to(self._model_device), examples[1]
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach().cpu()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        # logits_labels = [
        #     (self._logits_transformation(self._model(examples[0].to(self._model_device))).detach().cpu(),
        #      examples[1])
        #     for examples in tqdm(cal_dataloader)
        # ]
        # logits, labels = map(
        #     lambda x: torch.stack(x).float(),
        #     zip(*logits_labels)
        # )
        self.calculate_threshold(logits, labels, alpha)
        
        
        
    def calculate_threshold(self, logits, labels, alpha):
        
        self.scores = torch.zeros(logits.shape[0])
        for index, (x, y) in enumerate(zip(logits, labels)):
            self.scores[index] = self.score_function(x, y)
        self.q_hat = torch.quantile(self.scores, np.ceil((self.scores.shape[0] + 1) * (1 - alpha)) / self.scores.shape[0])
        


    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        if self._model != None:
            x_batch = self._model(x_batch.to(self._model_device)).float()
        x_batch = self._logits_transformation(x_batch).detach().cpu()
        sets = []
        for index, logits in enumerate(x_batch):
            sets.append(self.predict_with_logits(logits))
        return sets
    
    def predict_with_logits(self, logits):
        """ The input of score function is softmax probability.

        Args:
            probs (_type_): _description_

        Returns:
            _type_: _description_
        """
        scores = self.score_function.predict(logits)
        S = self._generate_prediction_set(scores, self.q_hat)
        return S
    
    
    #############################
    # The evaluation process
    ############################
    
    def evaluate(self, val_dataloader):
        prediction_sets = []
        labels_list = []
        with torch.no_grad():
                for  examples in tqdm(val_dataloader):
                    tmp_x, tmp_label = examples[0], examples[1]            
                    prediction_sets_batch = self.predict(tmp_x)
                    prediction_sets.extend(prediction_sets_batch)
                    labels_list.append(tmp_label)
        val_labels = torch.cat(labels_list)
        
        res_dict = {}
        res_dict["Coverage_rate"] = self._metric('coverage_rate')(prediction_sets, val_labels)
        res_dict["Average_size"] = self._metric('average_size')(prediction_sets, val_labels)
        return res_dict
    

    

        

    
    
    
