
import torch
import torch.nn as nn
class QuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize

        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)


        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss

        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)

        Returns
        -------
        loss : cost function value

        """
        assert not target.requires_grad
        if preds.size(0) != target.size(0):
            raise IndexError(f"Shape of preds must be equal to shape of target.")
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss