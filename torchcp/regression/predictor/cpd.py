# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.regression.score import Sign
from torchcp.regression.predictor.base import BasePredictor


class ConformalPredictiveDistribution(BasePredictor):
    """
    Obtain conformal predictive distributions from conformal predictive system.
        
    Args:
        score_function (torchcp.regression.scores): the score function must be Sign.
        model (torch.nn.Module): A pytorch regression model that can output predicted point.
        alpha (float, optional): The significance level. Default is 0.1.
        device (torch.device, optional): The device on which the model is located. Default is None.
        
    Reference:
        Paper: Algorithmic Learning in a Random World (vovk et al., 2022)
        Link: https://link.springer.com/book/10.1007/978-3-031-06649-8
    """

    def __init__(self, score_function=Sign(), model=None, alpha=0.1, device=None):
        super().__init__(score_function, model, alpha, device)

        if type(score_function) is not Sign:
            raise ValueError("score_function must be Sign().")

    def calculate_score(self, predicts, y_truth):
        """
        Calculate the nonconformity scores based on the model's predictions and true values.

        Args:
            predicts (torch.Tensor): Model predictions.
            y_truth (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed scores for each prediction.
        """
        return self.score_function(predicts, y_truth)

    def calibrate(self, cal_dataloader):
        self._model.eval()
        predicts_list, y_truth_list = [], []
        with torch.no_grad():
            for tmp_x, tmp_labels in cal_dataloader:
                tmp_x, tmp_labels = tmp_x.to(self._device), tmp_labels.to(self._device)
                tmp_predicts = self._model(tmp_x).detach()
                predicts_list.append(tmp_predicts)
                y_truth_list.append(tmp_labels)

        predicts = torch.cat(predicts_list).float().to(self._device)
        y_truth = torch.cat(y_truth_list).to(self._device)
        self.scores = self.calculate_score(predicts, y_truth)

    def predict(self, x_batch):
        """
        Obtain conformal predictive distributions from conformal predictive
        
        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            Tensor: conformal predictive distributions., shape (n_test, n_calib)
        """

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        x_batch = self._model(x_batch.to(self._device)).float()

        cpds = x_batch.unsqueeze(1) + self.scores.unsqueeze(0)
        return cpds.squeeze(-1)