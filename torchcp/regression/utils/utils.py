# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def calculate_midpoints(data_loader, K):
    """
    Calculates `K` evenly spaced midpoints between the minimum and maximum values 
    in a dataset loaded from `data_loader`.
    
    Used for R2CCP predictor.
    
    Args:
        data_loader (DataLoader): A PyTorch DataLoader containing tuples where the 
                                  second element (data[1]) is the tensor of interest.
        K (int): The number of midpoints to calculate.

    Returns:
        torch.Tensor: A tensor containing `K` midpoints evenly spaced between the minimum 
                      and maximum values of the concatenated dataset.
    """
    data_tensor = torch.cat([data[1] for data in data_loader], dim=0)
    midpoints = torch.linspace(data_tensor.min(), data_tensor.max(), steps=K)

    return midpoints
