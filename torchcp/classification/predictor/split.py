# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import torch
import warnings
from torch.utils.data import DataLoader
from typing import Dict, List

from torchcp.utils.common import calculate_conformal_value
from .base import BasePredictor


class SplitPredictor(BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
    """

    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)

    #############################
    # The calibration process
    ############################
    def calibrate(self, cal_dataloader, alpha):

        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

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
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha, marginal_q_hat=torch.inf):
        return calculate_conformal_value(scores, alpha, marginal_q_hat)

    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        x_batch = self._model(x_batch.to(self._device)).float()
        x_batch = self._logits_transformation(x_batch).detach()
        sets = self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        """
        Generate prediction sets from logits.

        Args:
            logits (torch.Tensor): Model output before softmax.
            q_hat (torch.Tensor, optional): The conformal threshold. Default is None.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        scores = self.score_function(logits).to(self._device)
        if q_hat is None:
            if self.q_hat is None:
                raise ValueError("Ensure self.q_hat is not None. Please perform calibration first.")
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)

        return S

    #############################
    # The evaluation process
    ############################

    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate prediction sets on validation dataset.
        
        Args:
            val_dataloader (torch.utils.data.DataLoader): Dataloader for validation set.
        
        Returns:
            dict: Dictionary containing evaluation metrics:
                - Coverage_rate: Empirical coverage rate on validation set
                - Average_size: Average size of prediction sets
        """
        predictions_sets_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        # Evaluate in inference mode
        self._model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device and get predictions
                inputs = batch[0].to(self._device)
                labels = batch[1].to(self._device)

                # Get predictions as bool tensor (N x C)
                batch_predictions = self.predict(inputs)

                # Accumulate predictions and labels
                predictions_sets_list.append(batch_predictions)
                labels_list.append(labels)

        # Concatenate all batches
        val_prediction_sets = torch.cat(predictions_sets_list, dim=0)  # (N_val x C)
        val_labels = torch.cat(labels_list, dim=0)  # (N_val,)

        # Compute evaluation metrics
        metrics = {
            "coverage_rate": self._metric('coverage_rate')(val_prediction_sets, val_labels),
            "average_size": self._metric('average_size')(val_prediction_sets, val_labels)
        }

        return metrics
