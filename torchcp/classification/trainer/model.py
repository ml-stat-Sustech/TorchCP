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
