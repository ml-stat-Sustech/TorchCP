# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def set_losses_from_labels(set_labels):
    """Given individual labels, compute set loss."""
    return torch.cumprod(1 - set_labels, axis=-1)
