# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable, Union, List

from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model import TemperatureScalingModel

class TSTrainer(Trainer):

    def __init__(
            self,
            model: torch.nn.Module,
            temperature: float,
            optimizer: torch.optim.Optimizer,
            loss_fn: Union[torch.nn.Module, Callable, List[Callable]],
            loss_weights: Optional[List[float]] = None,
            device: torch.device = None,
            verbose: bool = True,
            
    ):
        self.model = TemperatureScalingModel(model, temperature=temperature)
        # Create new optimizer instance for temperature only
        optimizer = type(optimizer)([self.model.temperature], **optimizer.defaults)
        super().__init__(self.model, optimizer, loss_fn, loss_weights, device, verbose)
        
        