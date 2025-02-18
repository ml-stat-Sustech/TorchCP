# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch import Tensor


class TemperatureScalingModel(nn.Module):
    """
    Model wrapper for temperature scaling in conformal prediction.
    
    Applies learnable temperature parameter to logits to improve
    calibration for conformal prediction.
    
    Args:
        base_model (nn.Module): Pre-trained model to be calibrated
        temperature (float): Initial temperature value (default: 1.0)
    
    Shape:
        - Input: Same as base_model input
        - Output: (batch_size, num_classes) scaled logits
    
    Examples:
        >>> base_model = resnet18(pretrained=True)
        >>> model = TemperatureScalingModel(base_model)
        >>> inputs = torch.randn(10, 3, 224, 224)
        >>> scaled_logits = model(inputs)
        
    Reference:
        Guo et al. "On Calibration of Modern Neural Networks", ICML 2017, https://arxiv.org/abs/1706.04599
    """

    def __init__(self, base_model: nn.Module, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

        # Freeze base model parameters
        self.freeze_base_model()

    def freeze_base_model(self):
        """Freeze all parameters in base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def is_base_model_frozen(self) -> bool:
        """Check if base model parameters are frozen"""
        return not any(p.requires_grad for p in self.base_model.parameters())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with temperature scaling.
        
        Args:
            x: Input tensor
            
        Returns:
            Temperature scaled logits
        """
        with torch.no_grad():  # Ensure no gradients flow through base model
            logits = self.base_model(x)

        return logits / self.temperature

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()

    def set_temperature(self, temp: float) -> None:
        """Set temperature value."""
        if temp <= 0:
            raise ValueError("Temperature must be positive")
        with torch.no_grad():
            self.temperature.fill_(temp)

    def train(self, mode: bool = True):
        """
        Override train method to ensure base_model stays in eval mode
        
        Args:
            mode: boolean to specify training mode
        """
        super().train(mode)  # Set training mode for TemperatureScalingModel
        self.base_model.eval()  # Keep base_model in eval mode
        return self


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
