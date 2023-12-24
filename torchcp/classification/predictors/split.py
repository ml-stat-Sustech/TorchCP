# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings
import math
import torch

from torchcp.classification.predictors.base import BasePredictor


class SplitPredictor(BasePredictor):
    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)


    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits,labels)
        # scores = logits.new_zeros(logits.shape[0])
        # for index, (x, y) in enumerate(zip(logits, labels)):
        #     scores[index] = self.score_function(x, y)
        self.q_hat = self._calculate_conformal_value(scores, alpha)
        
    def _calculate_conformal_value(self, scores, alpha):
        """
        Calculate the 1-alpha quantile of scores.
        
        :param scores: non-conformity scores.
        :param alpha: a significance level.
        
        :return: the threshold which is use to construct prediction sets.
        """
        if len(scores) == 0:
            warnings.warn("The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as torch.inf.")
            return torch.inf
        qunatile_value = math.ceil(scores.shape[0] + 1) * (1 - alpha) / scores.shape[0]
        
        if qunatile_value > 1:
            warnings.warn("The value of quantile exceeds 1. It should be a value in (0,1). To avoid program crash, the threshold is set as torch.inf.")
            return torch.inf
        
        return torch.quantile(scores, qunatile_value).to(self._device)

    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        """
        The input of score function is softmax probability.

        :param x_batch: a batch of instances.
        """
        self._model.eval()
        if self._model != None:
            x_batch = self._model(x_batch.to(self._device)).float()
        x_batch = self._logits_transformation(x_batch).detach()
        sets =  self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        """
        The input of score function is softmax probability.
        if q_hat is not given by the function 'self.calibrate', the construction progress of prediction set is a naive method.

        :param logits: model output before softmax.
        :param q_hat: the conformal threshold.

        :return: prediction sets
        """
        scores = self.score_function.predict(logits).to(self._device)
        if q_hat is None:
            S = self._generate_prediction_set(scores, self.q_hat)
        else:
            S = self._generate_prediction_set(scores, q_hat)
        return S

    #############################
    # The evaluation process
    ############################

    def evaluate(self, val_dataloader):
        prediction_sets = []
        labels_list = []
        with torch.no_grad():
            for examples in val_dataloader:
                tmp_x, tmp_label = examples[0].to(self._device), examples[1].to(self._device)
                prediction_sets_batch = self.predict(tmp_x)
                prediction_sets.extend(prediction_sets_batch)
                labels_list.append(tmp_label)
        val_labels = torch.cat(labels_list)

        res_dict = {}
        res_dict["Coverage_rate"] = self._metric('coverage_rate')(prediction_sets, val_labels)
        res_dict["Average_size"] = self._metric('average_size')(prediction_sets, val_labels)
        return res_dict
