# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# @Time : 15/12/2023  17:35


import torch

class ABS:
    def __init__(self):
        pass

    def __call__(self, predicts, labels):
        return torch.abs(predicts-labels)
