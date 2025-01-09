# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from torchcp.classification.trainer.base_trainer import BaseTrainer
from torchcp.classification.trainer.model import TemperatureScalingModel

class TSTrainer(BaseTrainer):
    """Temperature Scaling Trainer for model calibration.
    
    This trainer implements temperature scaling to calibrate neural network 
    predictions. It optimizes a single temperature parameter that divides the 
    logits to improve model calibration.
    
    Args:
        model (torch.nn.Module): Base neural network model to calibrate
        init_temperature (float): Initial temperature scaling parameter
        device (torch.device, optional): Device to run on. Defaults to None
        verbose (bool, optional): Whether to print progress. Defaults to True
        
    Attributes:
        model (TemperatureScalingModel): Model wrapped with temperature scaling
        device (torch.device): Device model is running on
        verbose (bool): Whether to print training progress
    """
    
    def __init__(
            self,
            model: torch.nn.Module,
            init_temperature: float,
            device: torch.device = None,
            verbose: bool = True):  
        
        self.init_temperature = init_temperature
        model = TemperatureScalingModel(model, temperature=init_temperature)        
        super().__init__(model, device, verbose)
        
    def train(
            self,
            train_loader: DataLoader,
            lr: float = 0.01,
            num_epochs: int = 100):
        """Train temperature scaling parameter using LBFGS optimizer.
        
        Collects logits and labels from training data, then optimizes the
        temperature parameter to minimize NLL loss.
        
        Args:
            train_loader (DataLoader): DataLoader with calibration data
            lr (float, optional): Learning rate for LBFGS. Defaults to 0.01
            num_epochs (int, optional): Max LBFGS iterations. Defaults to 100
        """
        self.model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # Collect logits and labels
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels.to(self.device))
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Calculate metrics before scaling
        before_nll = nll_criterion(logits, labels).item()
        before_ece = ece_criterion(logits, labels).item()
        
        if self.verbose:
            print(f'Before scaling - NLL: {before_nll:.3f}, ECE: {before_ece:.3f}')

        # Optimize temperature
        optimizer = optim.LBFGS([self.model.temperature], lr=lr, max_iter=num_epochs)

        def eval():
            optimizer.zero_grad()
            scaled_logits = logits / self.model.temperature
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate metrics after scaling
        after_nll = nll_criterion(logits / self.model.temperature, labels).item()
        after_ece = ece_criterion(logits / self.model.temperature, labels).item()
        
        if self.verbose:
            print(f'Optimal temperature: {self.model.temperature.item():.3f}')
            print(f'After scaling - NLL: {after_nll:.3f}, ECE: {after_ece:.3f}')
        return self.model
            
            
# Adapted from: Geoff Pleiss
# Source: https://github.com/gpleiss/temperature_scaling
# Original License: MIT.
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece