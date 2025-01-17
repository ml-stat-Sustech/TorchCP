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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable, Union, List

from torchcp.classification.trainer.base_trainer import Trainer


class OrdinalClassifier(nn.Module):
    """
    Ordinal Classifier 

    Args:
        classifier (nn.Module): A PyTorch classifier model.
        phi (str, optional): The transformation function for the classifier output. Default is "abs". Options are "abs" and "square".
        varphi (str, optional): The transformation function for the cumulative sum. Default is "abs". Options are "abs" and "square".

    Attributes:
        classifier (nn.Module): The classifier model.
        phi (str): The transformation function for the classifier output.
        varphi (str): The transformation function for the cumulative sum.
        phi_function (callable): The transformation function applied to the classifier output.
        varphi_function (callable): The transformation function applied to the cumulative sum.

    Methods:
        forward(x):
            Forward pass of the model.
    
    Examples::
        >>> classifier = nn.Linear(10, 5)
        >>> model = OrdinalClassifier(classifier, phi="abs", varphi="abs")
        >>> x = torch.randn(3, 10)
        >>> output = model(x)
        >>> print(output)
        
    Reference:
        DEY et al., 2023. "Conformal Prediction Sets for Ordinal Classification", https://proceedings.neurips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html
    """

    def __init__(self, classifier, phi="abs", varphi="abs"):
        super().__init__()
        self.classifier = classifier
        self.phi = phi
        self.varphi = varphi

        phi_options = {"abs": torch.abs, "square": torch.square}
        varphi_options = {"abs": lambda x: -torch.abs(x), "square": lambda x: -torch.square(x)}

        if phi not in phi_options:
            raise NotImplementedError(f"phi function '{self.phi}' is not implemented. Options are 'abs' and 'square'.")
        if varphi not in varphi_options:
            raise NotImplementedError(
                f"varphi function '{self.varphi}' is not implemented. Options are 'abs' and 'square'.")

        self.phi_function = phi_options[phi]
        self.varphi_function = varphi_options[phi]

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the classifier, phi_function, and varphi_function.
        """
        if x.shape[1] <= 2:
            raise ValueError("The input dimension must be greater than 2.")

        x = self.classifier(x)

        # a cumulative summation
        x = torch.cat((x[:, :1], self.phi_function(x[:, 1:])), dim=1)

        x = torch.cumsum(x, dim=1)

        # the unimodal distribution
        x = self.varphi_function(x)
        return x

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



