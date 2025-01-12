# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torchcp.classification.loss.confts import ConfTS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS
from torchcp.classification.trainer.base_trainer import BaseTrainer
from torchcp.classification.trainer.model import TemperatureScalingModel
from torchcp.classification.trainer.base_trainer import TrainingAlgorithm

class ConfTSTrainer(BaseTrainer):
    """Conformal Temperature Scaling Trainer.
    
    Args:
        model (torch.nn.Module): Base neural network model to be calibrated
        init_temperature (float): Initial value for temperature scaling parameter
        optimizer_class (torch.optim.Optimizer): Optimizer class for temperature parameter
            Default: torch.optim.Adam
        optimizer_params (dict): Parameters passed to optimizer constructor
            Default: {}
        device (torch.device): Device to run computations on 
            Default: None (auto-select GPU if available)
        verbose (bool): Whether to display training progress
            Default: True
        alpha (float): Target miscoverage rate (significance level) for conformal prediction
            Default: 0.1
            
    Examples:
        >>> # Initialize a CNN model
        >>> cnn = torchvision.models.resnet18(pretrained=True)
        >>> 
        >>> # Create ConfTS trainer
        >>> trainer = ConfTSTrainer(
        ...     model=cnn,
        ...     init_temperature=1.5,
        ...     optimizer_params={'lr': 0.01},
        ...     alpha=0.1
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
            optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_params: dict = {},
            device: torch.device = None,
            verbose: bool = True,
            alpha: float = 0.1):
        model = TemperatureScalingModel(model, temperature=init_temperature)
        super().__init__(model, device=device, verbose=verbose)
        optimizer = optimizer_class(
            [model.temperature],
            **optimizer_params
        )
        predictor = SplitPredictor(score_function=APS(score_type="softmax", randomized=False), model=model)
        confts = ConfTS(predictor=predictor, alpha=alpha, fraction=0.5)
        self.training_algorithm = TrainingAlgorithm(model=model, optimizer=optimizer, loss_fn=[confts], device=device, verbose=verbose)
        


        
        
