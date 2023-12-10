# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from abc import ABCMeta, abstractmethod



import torch
from tqdm import tqdm


class BasePredictor(object):
    """
    Abstract base class for all predictor classes.

    :param score_function: non-conformity score function.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function):
        """
        :calibration_method: methods used to calibrate 
        :param **kwargs: optional parameters used by child classes.
        """

        self.score_function = score_function

    @abstractmethod
    def calibrate(self,model, cal_dataloader, alpha):
        """Virtual method to calibrate the calibration set.

        :param model: the deep learning model.
        :param cal_dataloader : dataloader of calibration set.
        :param alpha: the significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """generate a prediction set for a test example.

        :param x: the model's output logits.
        """
        raise NotImplementedError
    
    def _cal_model_output(self,model, dataloader):
        self.model = model
        self.model_device = next(model.parameters()).device

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for  examples in tqdm(dataloader):
                tmp_x, tmp_label = examples[0].to(self.model_device), examples[1]            
                tmp_logits = model(tmp_x).detach().cpu()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_label)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
        return logits, labels
