import logging
import numpy as np
import time
import torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable, Union, List

from .base_trainer import Trainer


class OrdinalTrainer(Trainer):
    """
    OrdinalTrainer is a class for training ordinal classifiers.

    This class extends the Trainer class and provides methods for training, evaluating, and predicting with ordinal classifiers.
    It supports various configurations and training strategies to handle ordinal data.

    Args:
        model (torch.nn.Module): The backbone model to be trained.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        loss_fn (Union[torch.nn.Module, Callable, List[Callable]]): Loss function(s). Can be:
            - A PyTorch loss module
            - A custom loss function
            - A list of loss functions for multi-loss training.
        device (torch.device): Training device (CPU/GPU).
        verbose (bool, optional): Whether to print training progress and logs. Default is True.
        loss_weights (Optional[List[float]], optional): Weights for multiple loss functions if loss_fn is a list. Default is None.
        ordinal_config (Dict[str, Any], optional): Configuration for the ordinal classifier. Default is {"phi": "abs", "varphi": "abs"}.

    Attributes:
        model (OrdinalClassifier): The ordinal classifier model.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: Union[torch.nn.Module, Callable, List[Callable]],
            device: torch.device,
            verbose: bool = True,
            loss_weights: Optional[List[float]] = None,
            ordinal_config: Dict[str, Any] = {"phi": "abs", "varphi": "abs"}
    ):
        super().__init__(model, optimizer, loss_fn, device, verbose, loss_weights)
        self.model = OrdinalClassifier(model, **ordinal_config)


class OrdinalClassifier(nn.Module):
    """
    Method: Ordinal Classifier 
    Paper: Conformal Prediction Sets for Ordinal Classification (DEY et al., 2023)
    Link: https://proceedings.neurips.cc/paper_files/paper/2023/hash/029f699912bf3db747fe110948cc6169-Abstract-Conference.html

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
