# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

def calculate_midpoints(data_loader, K):
    data_tensor = torch.cat([data[0] for data in data_loader], dim=0)
    midpoints = torch.linspace(data_tensor.min(), data_tensor.max(), steps=K)

    return midpoints