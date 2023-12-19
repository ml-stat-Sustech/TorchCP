# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import random

import numpy as np
import torch

__all__ = ["fix_randomness"]


def fix_randomness(seed=0):
    # Fix randomness
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


class DimensionError(Exception):
    pass
