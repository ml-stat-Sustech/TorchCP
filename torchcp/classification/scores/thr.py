# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

from torchcp.classification.scores.base import BaseScore


class THR(BaseScore):
    """
    Threshold conformal predictors (Sadinle et al., 2016).
    paper : https://arxiv.org/abs/1609.00451.
    
    :param score_type: a transformation on logits. Default: "softmax". Optional: "softmax", "Identity", "log_softmax" or "log".
    """

    def __init__(self, score_type="softmax") -> None:
        
        super().__init__()
        self.score_type = score_type
        if score_type == "Identity":
            self.transform = lambda x: x
        elif score_type == "softmax":
            self.transform = lambda x: torch.softmax(x, dim=- 1)
        elif score_type == "log_softmax":
            self.transform = lambda x: torch.log_softmax(x, dim=-1)
        elif score_type == "log":
            self.transform = lambda x: torch.log(x)
        else:
            raise NotImplementedError

    def __call__(self, logits, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        temp_values = self.transform(logits)
        if label is None:
            return self.__calculate_all_label(temp_values)
        else:
            return self.__calculate_single_label(temp_values, label)

    def __calculate_single_label(self, temp_values, label):
        return 1 - temp_values[torch.arange(label.shape[0], device=temp_values.device), label]

    def __calculate_all_label(self, temp_values):
        return 1 - temp_values
