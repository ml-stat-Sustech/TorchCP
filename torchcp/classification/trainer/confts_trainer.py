# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from tqdm import tqdm
import torch

from torchcp.classification.loss.confts import ConfTS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS
from torchcp.classification.trainer.ts_trainer import TSTrainer
from torchcp.classification.trainer.model_zoo import TemperatureScalingModel


class ConfTSTrainer(TSTrainer):
    """Conformal Temperature Scaling Trainer.
    
    Args:
        init_temperature (float): Initial value for temperature scaling parameter.
        alpha (float): Target miscoverage rate (significance level) for conformal prediction.
        model (torch.nn.Module): Base neural network model to be calibrated.
        device (torch.device, optional): Device to run the model on. If None, will automatically use GPU ('cuda') if available, otherwise CPU ('cpu')
            Default: None
        verbose (bool): Whether to display training progress. Default: True.
        
            
    Examples:
        >>> # Initialize a CNN model
        >>> cnn = torchvision.models.resnet18(pretrained=True)
        >>> 
        >>> # Create ConfTS trainer
        >>> trainer = ConfTSTrainer(
        ...     init_temperature=1.5,
        ...     alpha=0.1    
        ...     model=cnn)
        >>> 
        >>> # Train calibration
        >>> trainer.train(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     num_epochs=10
        ... )
        >>> 
        >>> # Save calibrated model
        >>> trainer.save_model('calibrated_model.pth')
        """

    def __init__(
            self,
            alpha: float,
            init_temperature: float,
            model: torch.nn.Module,
            device: torch.device = None,
            verbose: bool = True, ):
        super().__init__(init_temperature, model, device=device, verbose=verbose)
        self.optimizer = torch.optim.Adam([self.model.temperature])
        predictor = SplitPredictor(score_function=APS(score_type="softmax", randomized=False), model=model)
        self.loss_fn = ConfTS(predictor=predictor, alpha=alpha, fraction=0.5)
        
        
    def train(self, train_loader, lr = 0.01, num_epochs = 100):
        for epoch in range(num_epochs):

            self.model.train()
            total_loss = 0

            # Create progress bar if verbose
            train_iter = tqdm(train_loader, desc="Training") if self.verbose else train_loader

            for data, target in train_iter:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if self.verbose:
                    train_iter.set_postfix({'loss': loss.item()})
        return  self.model
