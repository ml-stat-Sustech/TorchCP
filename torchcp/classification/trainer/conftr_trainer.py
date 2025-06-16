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
    """Conformal Training Trainer."""

    def __init__(
            self,
            alpha: float,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True):
        super().__init__(model, device=device, verbose=verbose)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        predictor = SplitPredictor(score_function=APS(score_type="softmax", randomized=False), model=model)
        self.loss_fn = ConfTrLoss(predictor=predictor, alpha=alpha, fraction=0.5)
        