# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tqdm import tqdm
import torch

from torchcp.classification.loss.scpo import SCPOLoss
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR, APS
from torchcp.classification.trainer.base_trainer import Trainer
from torchcp.classification.trainer.model_zoo import SurrogateCPModel


class SCPOTrainer(Trainer):
    """Surrogate Conformal Predictor Optimization.
    """

    def __init__(
            self,
            alpha: float,
            weight: float,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True, ):
        model = SurrogateCPModel(model)
        super().__init__(model, device=device, verbose=verbose)
        predictor = SplitPredictor(score_function=THR(score_type="identity"), model=model)
        self.loss_fn = SCPOLoss(predictor=predictor, alpha=alpha, fraction=0.5, weight=weight)
