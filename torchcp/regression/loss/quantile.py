import torch
import torch.nn as nn

__all__ = ["QuantileLoss"]


class QuantileLoss(nn.Module):
    r"""
    Pinball Loss (Quantile Loss) for Quantile Regression.
    This loss, also known as the pinball loss, is commonly used in quantile regression to estimate 
    the conditional quantiles of a target variable. It applies different penalties based on whether 
    the predictions fall above or below the actual target, making it useful for tasks requiring 
    interval or quantile estimation.

    Reference:
        Paper: Conformalized Quantile Regression (Romano, Y., et al., 2019)
        Link: https://proceedings.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf
        Github: https://github.com/yromano/cqr
        
    The quantile loss for each quantile level :math:`q \in (0, 1)` is defined as:

    .. math::
        L_q(y, \hat{y}) = 
        \begin{cases} 
            q \cdot (y - \hat{y}) & \text{if } y > \hat{y} \\
            (q - 1) \cdot (y - \hat{y}) & \text{if } y \leq \hat{y} 
        \end{cases}

    where:
    - :math:`y` is the target value,
    - :math:`\hat{y}` is the predicted quantile value.

    The total loss across all quantiles is averaged over the batch.

    Args:
        quantiles (list of float): List of quantiles to compute, typically in the range [0, 1],
            e.g., [0.025, 0.975] for the 2.5th and 97.5th percentiles.

    Shape:
        - Input: :attr:`preds` of shape `(batch_size, num_quantiles)` where `num_quantiles` is the number of specified quantiles.
        - Target: :attr:`target` of shape `(batch_size, 1)`.
        - Output: A scalar representing the mean quantile loss.

    Examples::

        >>> loss_fn = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        >>> preds = torch.rand(4, 3, requires_grad=True)
        >>> target = torch.rand(4, 1)
        >>> loss = loss_fn(preds, target)
        >>> loss.backward()
    """

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        Computes the mean pinball loss between predictions and targets.

        Args:
            preds (torch.Tensor): Predicted values for each quantile, shaped as `(batch_size, num_quantiles)`.
            target (torch.Tensor): Ground truth target values, shaped as `(batch_size, 1)`.

        Returns:
            torch.Tensor: The scalar mean pinball loss across the batch.

        Raises:
            AssertionError: If `target` requires gradients or if batch sizes of `preds` and `target` do not match.
        """
        assert not target.requires_grad, "Target should not require gradients."
        assert preds.size(0) == target.size(0), "Batch size mismatch between predictions and targets."

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            loss_q = torch.max((q - 1) * errors, q * errors).unsqueeze(1)
            losses.append(loss_q)

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
