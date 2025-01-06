# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

__all__ = ["HopCPTLoss"]


class HopCPTLoss(nn.Module):
    def __init__(self):
        super(HopCPTLoss, self).__init__()

    def forward(self, errors, association_matrix):
        weighted_errors = torch.matmul(association_matrix, errors.unsqueeze(-1)).squeeze(-1) 
        loss = torch.mean((torch.abs(errors) - weighted_errors) ** 2) 
        return loss
