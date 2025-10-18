# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from torchcp.regression.score.base import BaseScore


class RES(BaseScore):
    """
    RES score (Jin et al., 2023)
    paper: https://arxiv.org/pdf/2210.01408
    """
    def __call__(self, predicts, y_truth):
        if len(predicts.shape) == 2:
            predicts = predicts.squeeze().view(-1)
        return y_truth - predicts
