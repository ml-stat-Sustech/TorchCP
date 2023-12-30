import torch
import torch.nn as nn

__all__ = ["QuantileLoss", "R2ccpLoss"]


class QuantileLoss(nn.Module):
    """
    Pinball loss function (Romano et al., 2019).
    Paper: https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf

    :param quantiles: a list of quantiles, such as :math: [alpha/2, 1-alpha/2].
    """

    def __init__(self, quantiles):
        """

        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ 
        Compute the pinball loss.

        :param preds: the alpha/2 and 1-alpha/2 predictions of the model. The shape is batch x 2.
        :param target: the truth values. The shape is batch x 1.
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0), "the batch size of preds must be equal to the batch size of target."
        losses = preds.new_zeros(len(self.quantiles))

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i + 1]
            losses[i] = torch.sum(torch.max((q - 1) * errors, q * errors).squeeze(1))
        loss = torch.mean(losses)
        return loss
