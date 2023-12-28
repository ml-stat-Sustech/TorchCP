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
from torchcp.utils.common import calculate_conformal_value


class SplitPredictor(BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    :param score_function: non-conformity score function.
    :param model: a pytorch model.
    :param temperature: the temperature of Temperature Scaling.
    """
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
        scores = self.score_function(logits, labels)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

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
        sets = self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        """
        The input of score function is softmax probability.
        if q_hat is not given by the function 'self.calibrate', the construction progress of prediction set is a naive method.

        :param logits: model output before softmax.
        :param q_hat: the conformal threshold.

        :return: prediction sets
        """
        scores = self.score_function(logits).to(self._device)
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

        res_dict = {"Coverage_rate": self._metric('coverage_rate')(prediction_sets, val_labels),
                    "Average_size": self._metric('average_size')(prediction_sets, val_labels)}
        return res_dict
