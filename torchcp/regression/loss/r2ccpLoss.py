import torch
import torch.nn as nn

__all__ = ["R2ccpLoss"]
class R2ccpLoss(nn.Module):
    """
    Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2023).
    Paper: https://neurips.cc/virtual/2023/80610

    :param p: norm of distance measure.
    :param tau: weight of the ‘entropy’ term.
    :param midpoints: the midpoint of each bin.
    """

    def __init__(self, p, tau, midpoints):
        super().__init__()
        self.p = p
        self.tau = tau
        self.midpoints = midpoints

    def forward(self, preds, target):
        """ 
        Compute the cross-entropy loss with regularization

        :param preds: the softmax predictions of the model. The shape is batch*K.
        :param target: the truth values. The shape is batch*1.
        """
        assert not target.requires_grad
        if preds.size(0) != target.size(0):
            raise IndexError(f"Batch size of preds must be equal to the batch size of target.")
        
        target = target.view(-1, 1)
        abs_diff = torch.abs(target - self.midpoints.to(preds.device).unsqueeze(0))
        cross_entropy = torch.sum((abs_diff ** self.p) * preds, dim=1)
        shannon_entropy = torch.sum(preds * torch.log(preds.clamp_min(1e-10)), dim=1)
        
        losses = cross_entropy - self.tau * shannon_entropy
        loss = losses.sum()
        
        return loss
    