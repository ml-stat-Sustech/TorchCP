# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tqdm import tqdm
import torch

from torchcp.classification.loss.conftr import ConfTrLoss
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS
from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model_zoo import TemperatureScalingModel


class ConfTrTrainer(Trainer):
    """Conformal Training Trainer.

    Args:
        alpha (float): Target miscoverage rate (significance level) for conformal prediction.
        model (torch.nn.Module): Base neural network model to be calibrated.
        device (torch.device, optional): Device to run the model on. If None, will automatically use GPU ('cuda') if available, otherwise CPU ('cpu')
            Default: None
        verbose (bool): Whether to display training progress. Default: True.

        Examples:
            >>> # Initialize a CNN model
            >>> cnn = torchvision.models.resnet18(pretrained=True)
            >>> 
            >>> # Create ConfTr trainer
            >>> trainer = ConfTrTrainer(
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
    
        Reference:
            Stutz et al. "Learning Optimal Conformal Classifiers" (2022), https://arxiv.org/abs/2110.09192
    """

    def __init__(
            self,
            alpha: float,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True):
        super().__init__(model, device=device, verbose=verbose)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        predictor = SplitPredictor(score_function=APS(score_type="softmax", randomized=False), model=model)
        self.loss_fn = ConfTrLoss(predictor=predictor, alpha=alpha, fraction=0.5)
        