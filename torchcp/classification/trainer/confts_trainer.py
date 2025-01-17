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
from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model import TemperatureScalingModel


class ConfTSTrainer(Trainer):
    """Conformal Temperature Scaling Trainer.
    
    Args:
        init_temperature (float): Initial value for temperature scaling parameter.
        alpha (float): Target miscoverage rate (significance level) for conformal prediction.
        model (torch.nn.Module): Base neural network model to be calibrated.
        device (torch.device, optional): Device to run the model on. If None, will automatically use GPU ('cuda') if available, otherwise CPU ('cpu')
            Default: None
        verbose (bool): Whether to display training progress. Default: True.
        
            
    Examples:
        >>> # Initialize a CNN model
        >>> cnn = torchvision.models.resnet18(pretrained=True)
        >>> 
        >>> # Create ConfTS trainer
        >>> trainer = ConfTSTrainer(
        ...     init_temperature=1.5,
        ...     alpha=0.1    
        ...     model=cnn)
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
            init_temperature: float,
            alpha: float,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True, ):
        model = TemperatureScalingModel(model, temperature=init_temperature)
        super().__init__(model, device=device, verbose=verbose)
        self.optimizer = torch.optim.Adam([model.temperature])
        predictor = SplitPredictor(score_function=APS(score_type="softmax", randomized=False), model=model)
        self.loss_fn = ConfTS(predictor=predictor, alpha=alpha, fraction=0.5)

    def calculate_loss(self, output, target):
        """
        Calculate loss using multiple loss functions
        
        Args:
            output: Model output
            target: Ground truth
            
        Returns:
            Total loss value
        """
        return self.loss_fn(output, target)
