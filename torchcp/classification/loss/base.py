# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class BaseLoss(nn.Module):
    """
    Base class for conformal loss functions.

    Args:
        weight (float): The weight of the loss function. Must be greater than 0.
        predictor (object): An instance of a predictor class.
    """

    def __init__(self, weight, predictor, base_loss_fn=nn.CrossEntropyLoss()):
        super(BaseLoss, self).__init__()
        if weight <= 0:
            raise ValueError("weight must be greater than 0.")
        self.weight = weight
        self.predictor = predictor

    def forward(self, predictions, targets):
        raise NotImplementedError
