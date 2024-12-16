# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.trainer.ts_trainer import TSTrainer
from torchcp.classification.trainer.model import TemperatureScalingModel
from torchcp.classification.loss.confts import ConfTS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS

class ConfTSTrainer(TSTrainer):

    def __init__(
            self,
            model: torch.nn.Module,
            temperature: float,
            optimizer: torch.optim.Optimizer,
            device: torch.device = None,
            verbose: bool = True,
            alpha: float = 0.1
            
    ):
        
        self.model = TemperatureScalingModel(model, temperature=temperature)
        optimizer = type(optimizer)([self.model.temperature], **optimizer.defaults)
        
        predictor = SplitPredictor(score_function=APS(score_type="softmax", randomized=False), model=self.model)
        confts = ConfTS(predictor=predictor, alpha=alpha, fraction=0.5)
                
        super().__init__(model, temperature, optimizer, confts, device = device, verbose = verbose)
        
        