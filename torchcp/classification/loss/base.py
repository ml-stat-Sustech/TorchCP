
import torch.nn as nn

class BaseLoss(nn.Module):
    """
    Base class for conformal loss functions.

    Args:
        weight (float): The weight of the loss function. Must be greater than 0.
        predictor (object): An instance of a predictor class.
    """
    def __init__(self, weight, predictor, base_loss_fn=nn.CrossEntropyLoss()):
        super(BaseLoss, self).__init__()
        if weight <= 0:
            raise ValueError("weight must be greater than 0.")
        self.weight = weight
        self.predictor = predictor

    def forward(self, predictions, targets):
        raise NotImplementedError
