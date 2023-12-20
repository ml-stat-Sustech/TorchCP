import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """ Pinball loss function
    """

    def __init__(self, quantiles):
        """

       :param quantiles: quantile levels of predictions, each in the range (0,1)
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss

        :param preds: the alpha/2 and 1-alpha/2 predictions of the model
        :param target: the truth values
        """
        assert not target.requires_grad
        if preds.size(0) != target.size(0):
            raise IndexError(f"Shape of preds must be equal to shape of target.")
        losses = preds.new_zeros(len(self.quantiles))

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses[i] = torch.sum(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(losses)
        return loss
