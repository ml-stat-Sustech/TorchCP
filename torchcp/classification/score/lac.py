# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.score.base import BaseScore


class LAC(BaseScore):
    """
    Least Ambiguous Classifiers (LAC)
    
    Args:
        score_type (Union[str, Callable], optional): Specifies how to transform logits.
            - If str: Use predefined functions {"softmax", "identity", "log_softmax", "log"}
            - If callable: Custom function that takes and returns torch.Tensor
            Defaults to "softmax".

    Attributes:
        transform (callable): The transformation function applied to logits.
        
    Examples::
        >>> lac = LAC(score_type="softmax")
        >>> logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]])
        >>> scores_all = lac(logits)
        
        >>> # Using custom function
        >>> custom_transform = lambda x: x.sigmoid()
        >>> lac = LAC(score_type=custom_transform)
        >>> scores_custom = lac(logits)
        
    References:
        Sadinle, M. et al., (2016). Least ambiguous set-valued classifiers with bounded error levels. Journal of the American Statistical Association, 111(515), 1648-1658.
        
        Link : https://arxiv.org/abs/1609.00451
    """

    def __init__(self, score_type="softmax"):
        super().__init__()

        self.score_type = score_type

        if callable(score_type):
            self.transform = score_type
        else:
            if score_type == "identity":
                self.transform = lambda x: x
            elif score_type == "softmax":
                self.transform = lambda x: torch.softmax(x, dim=-1)
            elif score_type == "log_softmax":
                self.transform = lambda x: torch.log_softmax(x, dim=-1)
            elif score_type == "log":
                self.transform = lambda x: torch.log(x)
            else:
                raise ValueError(
                    f"Score type '{score_type}' is not implemented. Options are 'softmax', 'identity', 'log_softmax', 'log', or a callable function.")

    def __call__(self, logits, label=None):
        """
        Calculate non-conformity scores for logits.

        Args:
            logits (torch.Tensor): The logits output from the model.
            label (torch.Tensor, optional): The ground truth label. Default is None.

        Returns:
            torch.Tensor: The non-conformity scores.
        """

        if len(logits.shape) > 2:
            raise ValueError("dimension of logits are at most 2.")

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = self.transform(logits)
        if label is None:
            return self._calculate_all_label(probs)
        else:
            return self._calculate_single_label(probs, label)

    def _calculate_single_label(self, probs, label):
        """
        Calculate non-conformity score for a single label.

        Args:
            probs (torch.Tensor): The prediction probabilities.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The non-conformity score for the given label.
        """
        return 1 - probs[torch.arange(probs.shape[0], device=probs.device), label]

    def _calculate_all_label(self, probs):
        """
        Calculate non-conformity scores for all labels.

        Args:
            probs (torch.Tensor): The prediction probabilities.

        Returns:
            torch.Tensor: The non-conformity scores.
        """
        return 1 - probs
