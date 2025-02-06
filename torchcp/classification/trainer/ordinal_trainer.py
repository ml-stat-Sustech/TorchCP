# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict

import torch
import torch.nn as nn

from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model_zoo import OrdinalClassifier

class OrdinalTrainer(Trainer):
    """
    A trainer for training ordinal classifiers.

    This class extends the Trainer class and provides methods for training, evaluating, and predicting with ordinal classifiers.
    It supports various configurations and training strategies to handle ordinal data.

    Args:
        ordinal_config (Dict[str, str]): Configuration for ordinal classifier
            phi (str): Type of phi function ("abs", "square")
            varphi (str): Type of varphi function ("abs", "square")
            example: {"phi": "abs", "varphi": "abs"}
        model (torch.nn.Module): Base neural network model
        device (torch.device, optional): Device to run the model on. If None, will automatically use GPU ('cuda') if available, otherwise CPU ('cpu')
            Default: None
        verbose (bool): Whether to display training progress
            Default: True
            
    Examples:
        >>> # Define base model
        >>> backbone = torchvision.models.resnet18(pretrained=True)
        >>> 
        >>> # Configure ordinal classifier
        >>> ordinal_config = {
        ...     "phi": "square",
        ...     "varphi": "abs"
        ... }
        >>> 
        >>> # Create trainer
        >>> trainer = OrdinalTrainer(
        ...     ordinal_config=ordinal_config,
        ...     model=backbone)
        >>> 
        >>> # Train model
        >>> trainer.train(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=10
        ... )
    """

    def __init__(
            self,
            ordinal_config: Dict[str, str],
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True
    ):
        model = OrdinalClassifier(model, **ordinal_config)
        super().__init__(model, device, verbose)
        self.optimizer = torch.optim.Adam(model.parameters())
