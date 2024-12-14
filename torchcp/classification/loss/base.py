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
        predictor (torchcp.classification.Predictor): Predictor instance that defines
            the scoring mechanism for conformal prediction.
    """

    def __init__(self, predictor):
        super(BaseLoss, self).__init__()
        self.predictor = predictor

    def forward(self, predictions, targets):
        raise NotImplementedError
