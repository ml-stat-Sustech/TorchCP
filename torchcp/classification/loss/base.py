
import torch.nn as nn

class BaseLoss(nn.Module):
    
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        raise NotImplementedError("Forward method not implemented")
