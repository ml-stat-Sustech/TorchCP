# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# @Time : 13/12/2023  16:27
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
class ConfTr(nn.Module):
    def __init__(self, weights, predictor, alpha, device, fraction, loss_types = "valid", target_size = 1 , loss_transform = "square", base_loss_fn = None):
        """
        :param weights: the weight of each loss function
        :param predictor: the CP predictor
        :param alpha: the significance level for each training batch
        :param device: the device to use
        :param fraction: the fraction of the calibration set in each training batch
        :param types: the selected (multi-selected) loss functions, which can be "valid", "classification",  "probs", "coverage".
        :param target_size:
        :param loss_transform: a transform for loss
        :param loss_function: a base loss function, such as cross entropy for classification
        """
        super(ConfTr, self).__init__()
        self.weight = torch.tensor(weights).to(device)
        self.predictor = predictor
        self.alpha = alpha
        self.device = device
        self.fraction = fraction
        self.base_loss_fn = base_loss_fn
        
        self.target_size = target_size
        if loss_transform == "square":
            self.transform = torch.square
        elif loss_transform == "abs":
            self.transform = torch.abs
        elif loss_transform == "log":
            self.transform = torch.log
        else:
            raise NotImplementedError
        self.loss_functions_dict = {"valid": self.__compute_hinge_size_loss,
                                    "probs": self.__compute_probabilistic_size_loss,
                               "coverage": self.__compute_coverage_loss,
                               "classification": self.__compute_classification_loss}

        
        if type(loss_types) == set:
            if type(weights) != set:
                raise TypeError("weights must be a set.")
        elif type(loss_types) == str:
            if type(weights) != float and type(weights) != int:
                raise TypeError("weights must be a float or a int.")
        else: 
            raise TypeError("types must be a set or a string.")
        self.loss_types =  loss_types

    def forward(self, logits, labels):
        # Compute Size Loss
        val_split = int(self.fraction * logits.shape[0])
        cal_logits = logits[:val_split]
        cal_labels = labels[:val_split]
        test_logits = logits[val_split:]
        test_labels = labels[val_split:]

        self.predictor.calculate_threshold(cal_logits.detach(), cal_labels.detach(), self.alpha)
        tau = self.predictor.q_hat
        test_scores = self.predictor.score_function.predict(test_logits)
        pred_sets = torch.sigmoid(tau - test_scores)


        if type(self.loss_types) == set:
            loss = torch.tensor(0).to(self.device)
            for i in range(len(self.loss_types)):
                loss += self.weight[i] * self.loss_functions_dict[self.loss_types[i]](pred_sets, test_labels)
        else:
            loss = self.weight * self.loss_functions_dict[self.loss_types](pred_sets, test_labels)
            
        if self.base_loss_fn != None:
            loss += self.base_loss_fn(logits, labels).float()
       

        return  loss


    def __compute_hinge_size_loss(self,pred_sets, labels):
        return torch.mean(self.transform(torch.maximum(torch.sum(pred_sets, dim=1) - self.target_size, torch.tensor(0))))

    def __compute_probabilistic_size_loss(self, pred_sets, labels):
        classes = pred_sets.shape[0]
        one_hot_labels = torch.unsqueeze(torch.eye(classes),dim=0)
        repeated_confidence_sets = torch.repeat_interleave(
            torch.unsqueeze(pred_sets, 2), classes, dim=2)
        loss = one_hot_labels * repeated_confidence_sets + \
               (1 - one_hot_labels) * (1 - repeated_confidence_sets)
        loss = torch.prod(loss, dim=1)
        return torch.sum(loss, dim=1)


    def __compute_coverage_loss(self, pred_sets, labels):
        one_hot_labels = F.one_hot(labels, num_classes=pred_sets.shape[1])

        # Compute the mean of the sum of confidence_sets multiplied by one_hot_labels
        loss = torch.mean(torch.sum(pred_sets * one_hot_labels, dim=1)) - (1 - self.alpha)

        # Apply the transform function (you need to define this)
        transformed_loss = self.transform(loss)

        return transformed_loss

    def __compute_classification_loss(self, pred_sets, labels):
        # Convert labels to one-hot encoding
        one_hot_labels = F.one_hot(labels, num_classes=pred_sets.shape[1]).float()
        loss_matrix =  torch.eye(pred_sets.shape[1])
        # Calculate l1 and l2 losses
        l1 = (1 - pred_sets) * one_hot_labels * loss_matrix[labels]
        l2 = pred_sets * (1 - one_hot_labels) * loss_matrix[labels]

        # Calculate the total loss
        loss = torch.sum(torch.maximum(l1 + l2, torch.zeros_like(l1)), dim=1)

        # Return the mean loss
        return torch.mean(loss)



