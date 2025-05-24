# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.loss.scpo import SCPOLoss
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import LAC
from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model_zoo import SurrogateCPModel


class SCPOTrainer(Trainer):
    """
    Trainer for Surrogate Conformal Predictor Optimization.

    Args:
        alpha (float): The significance level for each training batch.
        model (torch.nn.Module): Base neural network model to be calibrated.
        device (torch.device, optional): Device to run the model on. If None, will automatically use GPU ('cuda') if available, otherwise CPU ('cpu')
            Default: None
        verbose (bool): Whether to display training progress. Default: True.
        lr (float): Learning rate for the optimizer. Default is 0.1.
        lambda_val (float): Weight for the coverage loss term.
        gamma_val (float): Inverse of the temperature value.
    
    Examples:
        >>> # Define base model
        >>> backbone = torchvision.models.resnet18(pretrained=True)
        >>> 
        >>> # Create SCPO trainer
        >>> trainer = SCPOTrainer(
        ...             alpha=0.01,
        ...             model=model,
        ...             device=device,
        ...             verbose=True)
        >>> 
        >>> # Train model
        >>> trainer.train(
        ...     train_loader=train_loader,
        ...     num_epochs=10
        ... )
    """

    def __init__(
            self,
            alpha: float,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True,
            lr: float = 0.1,
            lambda_val: float = 10000,
            gamma_val: float = 1):

        model = SurrogateCPModel(model)
        super().__init__(model, device=device, verbose=verbose)
        predictor = SplitPredictor(score_function=LAC(score_type="identity"), model=model)

        self.optimizer = torch.optim.Adam(self.model.linear.parameters(), lr=lr)
        self.loss_fn = SCPOLoss(predictor=predictor, alpha=alpha, 
                                lambda_val=lambda_val, gamma_val=gamma_val)
