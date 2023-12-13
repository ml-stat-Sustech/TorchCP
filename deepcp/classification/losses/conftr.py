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
    def __init__(self, weight, loss_cunftion, predictor, alpha, device, fraction, type = "valid", target_size = 1 , size_transform = "square"):
        super(ConfTr, self).__init__()
        self.loss_cunftion = loss_cunftion
        self.predictor = predictor
        self.target_size = target_size
        self.transform = torch.square
        self.weight = torch.tensor(weight).to(device)
        self.fraction = fraction
        self.alpha = alpha

    def forward(self, logits, labels):
        loss1 = self.loss_cunftion(logits, labels)

        # Compute Size Loss
        probs = F.softmax(logits,dim=1)
        val_split = int(self.fraction * probs.shape[0])
        val_probs = probs[:val_split]
        val_labels = labels[:val_split]
        test_probs = probs[val_split:]
        test_labels = labels[val_split:]

        self.predictor.calculate_threshold(val_probs.detach().cpu().numpy(), labels.detach().cpu().numpy(), self.alpha)
        tau = self.predictor.q_hat
        pred_sets = torch.sigmoid(tau - test_probs)
        
        size_loss = self.__compute_hinge_size_loss(pred_sets)

        return self.weight * size_loss + loss1


    def __compute_hinge_size_loss(self,pred_sets):
        return torch.mean(self.transform(torch.maximum(torch.sum(pred_sets, dim=1) - self.target_size, torch.tensor(0))))


