# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torchcp.regression.score.base import BaseScore


class CLIP(BaseScore):
    """
    CLIP score (Jin et al., 2023), only apply to binary classification.
    paper: https://arxiv.org/pdf/2210.01408
    """
    def __call__(self, predicts, y_truth, M=100):
        if len(predicts.shape) == 2:
            predicts = predicts.squeeze().view(-1)
        return M * torch.max(predicts, 0) - predicts
