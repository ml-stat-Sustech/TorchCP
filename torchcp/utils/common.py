# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import random

import numpy as np
import torch

__all__ = ["fix_randomness", "DimensionError", "get_device"]


def fix_randomness(seed=0):
    """
    Fix the random seed for python, torch, numpy.

    :param seed: the random seed
    """
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


class DimensionError(Exception):
    pass


def get_device(model):
    """
    Get the device of Torch model.

    :param model: a Pytorch model. If None, it uses GPU when the cuda is available, otherwise it uses CPU。

    :return: the device in use
    """
    if model is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = next(model.parameters()).device
    return device
