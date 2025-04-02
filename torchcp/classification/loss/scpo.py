# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__all__ = ["SCPO"]

import torch
from torch import Tensor

from torchcp.classification.loss.confts import ConfTSLoss
from torchcp.classification.loss.conftr import ConfTrLoss


class SCPOLoss(ConfTSLoss):

    def __init__(self, predictor, alpha, fraction=0.5, soft_qunatile=True, weight=5):

        super(SCPOLoss, self).__init__(predictor, alpha, fraction, soft_qunatile)
        self.weight = weight
        self.size_loss_fn = ConfTrLoss(predictor=predictor, alpha=alpha, fraction=0.5, loss_type="valid", target_size=0, loss_transform="abs")
        self.coverage_loss_fn = ConfTrLoss(predictor=predictor, alpha=alpha, fraction=0.5, loss_type="coverage")

    def compute_loss(self, test_scores, test_labels, tau):
        return torch.log(self.size_loss_fn.compute_loss(test_scores, test_labels, 1) \
            + self.weight * self.coverage_loss_fn.compute_loss(test_scores, test_labels, 1))