import torch
import torch.nn as nn


class R2ccpLoss(nn.Module):
    """
    Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2023).
    Paper: https://dev.neurips.cc/virtual/2023/80610

    :param p: norm of distance measure.
    :param tau: weighting term of the regularization.
    :param K: number of bins.
    """

    def __init__(self, p, tau, K):
        super().__init__()
        self.p = p
        self.tau = tau
        self.K = K

    def forward(self, preds, target):
        """ 
        Compute the cross-entropy loss with regularization

        :param preds: the softmax predictions of the model. The shape is batch*K.
        :param target: the truth values. The shape is batch*1.
        """
        assert not target.requires_grad
        if preds.size(0) != target.size(0):
            raise IndexError(f"Shape of preds must be equal to shape of target.")
        losses = preds.new_zeros(len(target))

        errors = torch.abs(target - preds)
        cross_entropy = torch.sum(errors ** self.p * target, dim=1)
        shannon_entropy = torch.sum(preds * torch.log(preds.clamp_min(1e-10)), dim=1)
        losses = cross_entropy - self.tau * shannon_entropy
        
        loss = losses.sum()
        # print(loss)
        return loss
    