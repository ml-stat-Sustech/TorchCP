# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import math
import torch

from torchcp.regression.utils.metrics import Metrics
from torchcp.utils.common import get_device
from torchcp.utils.common import calculate_conformal_value


class SplitPredictor(object):
    """
    Distribution-Free Predictive Inference For Regression (Lei et al., 2017)
    paper: https://arxiv.org/abs/1604.04173
    
    :param model: a pytorch model for regression.
    """

    def __init__(self, model):
        self._model = model
        self._device = get_device(model)
        self._metric = Metrics()
        
    def calculate_score(self, predicts, y_truth):
        if len(y_truth.shape) == 1:
            y_truth = y_truth.unsqueeze(1)
        return torch.abs(predicts - y_truth)

    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        predicts_list = []
        y_truth_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_predicts = self._model(tmp_x).detach()
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)
            predicts = torch.cat(predicts_list).float().to(self._device)
            y_truth = torch.cat(y_truth_list).to(self._device)
        self.calculate_threshold(predicts, y_truth, alpha)

    def calculate_threshold(self, predicts, y_truth, alpha):
        scores = self.calculate_score(predicts, y_truth)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

        
    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    def predict(self, x_batch):
        self._model.eval()
        x_batch.to(self._device)
        with torch.no_grad():
            predicts_batch = self._model(x_batch)
            prediction_intervals = x_batch.new_zeros((predicts_batch.shape[0],self.q_hat.shape[0] , 2))

            prediction_intervals[..., 0] = predicts_batch - self.q_hat.view(1, self.q_hat.shape[0])
            prediction_intervals[..., 1] = predicts_batch + self.q_hat.view(1, self.q_hat.shape[0])

        return prediction_intervals

    def evaluate(self, data_loader):
        y_list = []
        predict_list = []
        with torch.no_grad():
            for examples in data_loader:
                tmp_x, tmp_y = examples[0].to(self._device), examples[1].to(self._device)
                tmp_prediction_intervals = self.predict(tmp_x)
                y_list.append(tmp_y)
                predict_list.append(tmp_prediction_intervals)
                
        

        predicts = torch.cat(predict_list,dim=0).to(self._device)
        test_y = torch.cat(y_list).to(self._device)


        res_dict = {"Coverage_rate": self._metric('coverage_rate')(predicts, test_y),
                    "Average_size": self._metric('average_size')(predicts)}
        return res_dict
