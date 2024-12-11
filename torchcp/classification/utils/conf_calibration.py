# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim

from torchcp.utils.registry import Registry

ConfCalibrator_REGISTRY = Registry("ConfCalibrator")


class ConfCalibrator:
    @classmethod
    def registry_ConfCalibrator(cls, conf_calibrator):
        if conf_calibrator not in ConfCalibrator_REGISTRY.registered_names():
            raise NameError(f"The Confidence Calibrator: {conf_calibrator} is not defined in TorchCP.")
        return ConfCalibrator_REGISTRY.get(conf_calibrator)


@ConfCalibrator_REGISTRY.register()
class Identity(nn.Module):
    def forward(self, batch_logits):
        return batch_logits


@ConfCalibrator_REGISTRY.register()
class TS(nn.Module):
    """Using a pre-defiend tempreature to scale the logits"""

    def __init__(self, temperature=1) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, batch_logits):
        return batch_logits / self.temperature

    def optimze(self, dataloader, device, max_iters=10, lr=0.01, epsilon=0.01):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.SGD([self.temperature], lr=lr)

        for iter in range(max_iters):
            T_old = self.temperature.item()
            for x, targets in dataloader:
                optimizer.zero_grad()
                x = x.to(device)
                x.requires_grad = True
                out = x / self.temperature
                loss = nll_criterion(out, targets.long().cuda())

                loss.backward()
                optimizer.step()
            if abs(T_old - self.temperature.item()) < epsilon:
                break
