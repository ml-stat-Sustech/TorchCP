# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import time

from torchcp.classification.loss.confts import ConfTS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS
from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model import TemperatureScalingModel

class ConfTSTrainer(Trainer):
    """Conformal Temperature Scaling Trainer.
    
    Args:
        model (torch.nn.Module): Base neural network model to be calibrated
        init_temperature (float): Initial value for temperature scaling parameter
        alpha (float): Target miscoverage rate (significance level) for conformal prediction
                Default: 0.1
        optimizer_class (torch.optim.Optimizer): Optimizer class for temperature parameter
                Default: torch.optim.Adam
        optimizer_params (dict): Parameters passed to optimizer constructor
                Default: {}
        device (torch.device): Device to run computations on 
                Default: None (auto-select GPU if available)
        verbose (bool): Whether to display training progress
                Default: True
        
            
    Examples:
        >>> # Initialize a CNN model
        >>> cnn = torchvision.models.resnet18(pretrained=True)
        >>> 
        >>> # Create ConfTS trainer
        >>> trainer = ConfTSTrainer(
        ...     model=cnn,
        ...     init_temperature=1.5,
        ...     alpha=0.1
        ...     optimizer_params={'lr': 0.01},
        ... )
        >>> 
        >>> # Train calibration
        >>> trainer.train(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=10
        ... )
        >>> 
        >>> # Save calibrated model
        >>> trainer.save_model('calibrated_model.pth')
        """
    def __init__(
            self,
            model: torch.nn.Module,
            init_temperature: float,
            alpha: float = 0.1,
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_params: dict = {},
            device: torch.device = None,
            verbose: bool = True,):
        model = TemperatureScalingModel(model, temperature=init_temperature)
        super().__init__(model, device=device, verbose=verbose)
        self.optimizer = optimizer_class(
            [model.temperature],
            **optimizer_params
        )
        predictor = SplitPredictor(score_function=APS(score_type="softmax", randomized=False), model=model)
        self.loss_fns = [ConfTS(predictor=predictor, alpha=alpha, fraction=0.5)]
        self.loss_weights = [1.0]
    

        


        
        
