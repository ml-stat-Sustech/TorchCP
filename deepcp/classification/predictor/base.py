# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta, abstractmethod

import torch

from deepcp.classification.utils.metrics import Metrics
from deepcp.classification.utils import ConfCalibrator

class BasePredictor(object):
    """
    Abstract base class for all predictor classes.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model= None):
        """
        :score_function: non-conformity score function.
        :param model: a deep learning model.
        """

        self.score_function = score_function
        self._model = model
        if self._model ==  None:
            self._model_device = None
        else:
            self._model_device = next(model.parameters()).device
        self._metric = Metrics()
        self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("Identity")()

    @abstractmethod
    def calibrate(self, cal_dataloader, alpha):
        """Virtual method to calibrate the calibration set.

        :param cal_dataloader : a dataloader of calibration set.
        :param alpha: the significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_batch):
        """generate prediction sets for  test examples.
        :param x_batch: a batch of input.
        """
        raise NotImplementedError
    
        
    
    def _generate_prediction_set(self, scores, q_hat):
        """Generate the prediction set with the threshold q_hat.

        Args:
            scores (_type_): The non-conformity scores of {(x,y_1),..., (x,y_K)}
            q_hat (_type_): the calibrated threshold.

        Returns:
            _type_: _description_
        """
        if len(scores.shape) ==1:
            return torch.argwhere(scores < q_hat).reshape(-1).tolist()
        else:
            return torch.argwhere(scores < q_hat).tolist()


class InductivePredictor(BasePredictor):
    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model)

        #############################

    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._model_device), examples[1]
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):

        self.scores = torch.zeros(logits.shape[0])
        for index, (x, y) in enumerate(zip(logits, labels)):
            self.scores[index] = self.score_function(x, y)
        self.q_hat = torch.quantile(self.scores,
                                    torch.ceil(torch.tensor((self.scores.shape[0] + 1) * (1 - alpha))) / self.scores.shape[0])
        
    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        if self._model != None:
            x_batch = self._model(x_batch.to(self._model_device)).float()
        x_batch = self._logits_transformation(x_batch).detach()
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
            for examples in val_dataloader:
                tmp_x, tmp_label = examples[0], examples[1]
                prediction_sets_batch = self.predict(tmp_x)
                prediction_sets.extend(prediction_sets_batch)
                labels_list.append(tmp_label)
        val_labels = torch.cat(labels_list)

        res_dict = {}
        res_dict["Coverage_rate"] = self._metric('coverage_rate')(prediction_sets, val_labels)
        res_dict["Average_size"] = self._metric('average_size')(prediction_sets, val_labels)
        return res_dict
