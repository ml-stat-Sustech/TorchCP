# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import scipy

from deepcp.classification.scores.base import BaseScoreFunction


class THR(BaseScoreFunction):
    def __init__(self, score_type = "logits") -> None:
        """_summary_

        Args:
            score_type (str, optional): _description_. Defaults to "logits". Other values can be "logp" or "prob"
        """
        super().__init__()
        if score_type == "Identity":
            self.transform = lambda x: x
        elif score_type == "prob":
            self.transform = scipy.special.softmax
        elif score_type == "logp":
            self.transform = lambda x: np.log(scipy.special.softmax(x))
        elif score_type == "log":
            self.transform = lambda x: np.log(x)
        else:
            raise NotImplementedError

    def __call__(self, probs, y):
        return 1 - self.transform(probs)[y]

    def predict(self, probs):
        return 1 - self.transform(probs)
