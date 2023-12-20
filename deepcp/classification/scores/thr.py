# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from deepcp.classification.scores.base import BaseScoreFunction


class THR(BaseScoreFunction):
    """
    Threshold conformal predictor (Sadinle et al., 2016)
    paper : https://arxiv.org/abs/1609.00451
    """

    def __init__(self, score_type="softmax") -> None:
        """
        param score_type: either "softmax" "Identity", "log_softmax" or "log". Default: "softmax". A transformation for logits.
        """
        super().__init__()
        self.score_type = score_type
        if score_type == "Identity":
            self.transform = lambda x: x
        elif score_type == "softmax":
            self.transform = lambda x: torch.softmax(x, dim=len(x.shape) - 1)
        elif score_type == "log_softmax":
            self.transform = lambda x: torch.log_softmax(x, dim=len(x.shape) - 1)
        elif score_type == "log":
            self.transform = lambda x: torch.log(x, dim=len(x.shape) - 1)
        else:
            raise NotImplementedError

    def __call__(self, logits, y):
        if len(logits.shape) > 1:
            return 1 - self.transform(logits)[torch.arange(y.shape[0]).to(logits.device), y]
        else:
            return 1 - self.transform(logits)[y]

    def predict(self, logits):
        return 1 - self.transform(logits)
