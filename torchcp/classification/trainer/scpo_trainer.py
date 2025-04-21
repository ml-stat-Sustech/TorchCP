# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from torchcp.classification.loss.scpo import SCPOLoss
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR
from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model_zoo import SurrogateCPModel


class SCPOTrainer(Trainer):
    """Surrogate Conformal Predictor Optimization.
    """

    def __init__(
            self,
            alpha: float,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True,
            lr: float = 1000,
            lambda_val: float = 10000,
            gamma_val: float = 1):

        model = SurrogateCPModel(model)
        super().__init__(model, device=device, verbose=verbose)
        predictor = SplitPredictor(score_function=THR(score_type="identity"), model=model)

        self.optimizer = torch.optim.Adam(self.model.linear.parameters(), lr=lr)
        self.loss_fn = SCPOLoss(predictor=predictor, alpha=alpha, 
                                lambda_val=lambda_val, gamma_val=gamma_val)
