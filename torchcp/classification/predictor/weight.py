# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchcp.classification.predictor.split import SplitPredictor
from torchcp.classification.predictor.utils import build_DomainDetecor, IW


class WeightedPredictor(SplitPredictor):
    """
    Method: Weighted Conformal Prediction
    Paper: Conformal Prediction Under Covariate Shift (Tibshirani et al., 2019)
    Link: https://arxiv.org/abs/1904.06019
    Github: https://github.com/ryantibs/conformal/
    
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module): A PyTorch model.
        alpha (float, optional): The significance level. Default is 0.1.
        image_encoder (torch.nn.Module): A PyTorch model to generate the embedding feature of an input image.
        domain_classifier (torch.nn.Module, optional): A PyTorch model (a binary classifier) to predict the probability that an embedding feature comes from the source domain. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
        device (torch.device, optional): The device on which the model is located. Default is None.
    """

    def __init__(self, score_function, model=None, temperature=1, alpha=0.1, image_encoder=None, domain_classifier=None, device=None):

        super().__init__(score_function, model, temperature, alpha, device)

        if image_encoder is None:
            raise ValueError("image_encoder cannot be None.")

        self.image_encoder = image_encoder.to(self._device)
        self.domain_classifier = domain_classifier

        #  non-conformity scores
        self.scores = None
        # significance level
        self.alpha = None
        # Domain Classifier

    def calibrate(self, cal_dataloader, alpha=None):
        """
        Calibrate the model using the calibration set.

        Args:
            cal_dataloader (torch.utils.data.DataLoader): A dataloader of the calibration set.
            alpha (float): The significance level. Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        logits_list = []
        labels_list = []
        cal_features_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                cal_features_list.append(self.image_encoder(tmp_x))
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
            self.source_image_features = torch.cat(cal_features_list).float()

        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha=None):
        """
        Calculate the conformal prediction threshold.

        Args:
            logits (torch.Tensor): The logits output from the model.
            labels (torch.Tensor): The ground truth labels.
            alpha (float): The significance level. Default is None.
        """
        if alpha is None:
            alpha = self.alpha

        self.alpha = alpha
        self.scores = torch.zeros(logits.shape[0] + 1).to(self._device)
        self.scores[:logits.shape[0]] = self.score_function(logits, labels)
        self.scores[logits.shape[0]] = torch.tensor(torch.inf).to(self._device)
        self.scores_sorted = self.scores.sort()[0]

    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """
        if not hasattr(self, "scores_sorted"):
            raise ValueError("Please calibrate first to get self.scores_sorted.")

        bs = x_batch.shape[0]
        with torch.no_grad():
            image_features = self.image_encoder(x_batch.to(self._device)).float()
            w_new = self.IW(image_features)

        w_sorted = self.w_sorted.expand([bs, -1])
        w_sorted = torch.cat([w_sorted, w_new.unsqueeze(1)], 1)
        p_sorted = w_sorted / w_sorted.sum(1, keepdim=True)
        p_sorted_acc = p_sorted.cumsum(1)

        i_T = torch.argmax((p_sorted_acc >= 1.0 - self.alpha).int(), dim=1, keepdim=True)
        q_hat_batch = self.scores_sorted.expand([bs, -1]).gather(1, i_T).detach()

        logits = self._model(x_batch.to(self._device)).float()
        logits = self._logits_transformation(logits).detach()
        predictions_sets_list = []
        for index, (logits_instance, q_hat) in enumerate(zip(logits, q_hat_batch)):
            predictions_sets_list.append(self.predict_with_logits(logits_instance, q_hat))

        predictions_sets = torch.cat(predictions_sets_list, dim=0)  # (N_val x C)
        return predictions_sets

    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate prediction sets on validation dataset using domain adaptation.

        This method trains a domain classifier if not provided, computes importance 
        weights for validation set, generates prediction sets and calculates metrics.

        Args:
            val_dataloader (DataLoader): Dataloader for validation set.

        Returns:
            dict: Dictionary containing evaluation metrics:
                - Coverage_rate: Empirical coverage rate on validation set
                - Average_size: Average size of prediction sets

        Raises:
            ValueError: If calibration has not been performed first.
        """
        # Extract features from validation set
        self._model.eval()
        features_list: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch[0].to(self._device)
                features = self.image_encoder(inputs)
                features_list.append(features)
        target_features = torch.cat(features_list, dim=0).float()  # (N_val x D)

        # Train domain classifier if needed
        if not hasattr(self, "source_image_features"):
            raise ValueError("Please calibrate first to get source_image_features.")

        if self.domain_classifier is None:
            self._train_domain_classifier(target_features)

        # Compute importance weights
        self.IW = IW(self.domain_classifier).to(self._device)
        weights_cal = self.IW(self.source_image_features.to(self._device))
        self.w_sorted = torch.sort(weights_cal, descending=False)[0]

        # Generate predictions
        predictions_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device and get predictions
                inputs = batch[0].to(self._device)
                labels = batch[1].to(self._device)

                # Get predictions as bool tensor (N x C)
                batch_predictions = self.predict(inputs)

                # Accumulate predictions and labels
                predictions_list.append(batch_predictions)
                labels_list.append(labels)

        # Concatenate all batches
        val_predictions = torch.cat(predictions_list, dim=0)  # (N_val x C) 
        val_labels = torch.cat(labels_list, dim=0)  # (N_val,)

        # Compute evaluation metrics
        metrics = {
            "coverage_rate": self._metric('coverage_rate')(val_predictions, val_labels),
            "average_size": self._metric('average_size')(val_predictions, val_labels)
        }

        return metrics

    def _train_domain_classifier(self, target_image_features):
        source_labels = torch.zeros(self.source_image_features.shape[0]).to(self._device)
        target_labels = torch.ones(target_image_features.shape[0]).to(self._device)

        input = torch.cat((self.source_image_features, target_image_features))
        labels = torch.cat((source_labels, target_labels))
        dataset = torch.utils.data.TensorDataset(input.float(), labels.float().long())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=False)

        self.domain_classifier = build_DomainDetecor(target_image_features.shape[1], 2, self._device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.domain_classifier.parameters(), lr=0.001)

        epochs = 5
        for epoch in range(epochs):
            loss_log = 0
            accuracy_log = 0
            for X_train, y_train in data_loader:
                y_train = y_train.to(self._device)
                outputs = self.domain_classifier(X_train.to(self._device))
                loss = criterion(outputs, y_train.view(-1))
                loss_log += loss.item() / len(data_loader)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = torch.sum((predictions == y_train.view(-1))).item() / len(y_train)
                accuracy_log += accuracy / len(data_loader)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
