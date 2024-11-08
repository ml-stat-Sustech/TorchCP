import torch
import torch.nn as nn

__all__ = ["R2ccpLoss"]


class R2ccpLoss(nn.Module):
    r"""
    Conformal Prediction via Regression-as-Classification Loss.
    This loss combines cross-entropy and entropy regularization to provide uncertainty estimates for predictions, 
    supporting conformal prediction in regression tasks by treating it as a classification problem.
    
    Args:
        p (float): The norm degree for the distance measure in the cross-entropy term.
        tau (float): Weighting factor for the entropy regularization term.
        midpoints (torch.Tensor): A tensor containing midpoint values for each bin.

    Shape:
        - Input: :attr:`preds` of shape `(batch_size, K)` where `K` is the number of bins.
        - Target: :attr:`target` of shape `(batch_size, 1)`.
        - Output: A scalar representing the computed loss.

    Reference:
        Paper: Conformal Prediction via Regression-as-Classification (Etash Guha et al., 2021)
        Link: https://neurips.cc/virtual/2023/80610
        Github: https://github.com/EtashGuha/R2CCP

    The loss is composed of two main parts:
    
    1. Cross-entropy component:
       This term computes the weighted distance between the prediction probabilities and target midpoint values 
       based on a specified distance norm :attr:`p`.

    2. Entropy regularization term:
       This term applies a Shannon entropy penalty, controlled by the hyperparameter :attr:`tau`, 
       which helps balance between prediction certainty and regularization for the model's output distribution.

    The total loss is given by:

    .. math::
        \mathcal{L} = \sum_{i=1}^N \left( \sum_{k=1}^K |y_i - m_k|^p \cdot \hat{p}_{ik} - \tau \sum_{k=1}^K \hat{p}_{ik} \log(\hat{p}_{ik}) \right)

    where:
    - :math:`y_i` is the target for instance `i`,
    - :math:`m_k` is the midpoint for bin `k`,
    - :math:`\hat{p}_{ik}` is the predicted probability for bin `k` and instance `i`,
    - :math:`\tau` is the weight for the entropy regularization term,
    - :math:`p` is the norm for the distance measure.

    Examples::

        >>> loss_fn = R2ccpLoss(p=2, tau=0.5, midpoints=torch.tensor([0.1, 0.5, 0.9]))
        >>> preds = torch.rand(3, 3, requires_grad=True).softmax(dim=1)
        >>> target = torch.tensor([[0.2], [0.6], [0.8]])
        >>> loss = loss_fn(preds, target)
        >>> loss.backward()
    """
    
    def __init__(self, p, tau, midpoints):
        super().__init__()
        self.p = p
        self.tau = tau
        self.midpoints = midpoints

    def forward(self, preds, target):
        """
        Computes the R2ccp loss between model predictions and targets.

        Args:
            preds (torch.Tensor): Model predictions after applying softmax, shape `(batch_size, K)`.
            target (torch.Tensor): Ground truth values, shape `(batch_size, 1)`.

        Returns:
            torch.Tensor: Scalar loss value.

        Raises:
            AssertionError: If `target` requires gradients.
            IndexError: If batch size of `preds` does not match `target`.
        """
        assert not target.requires_grad, "Target should not require gradients."
        if preds.size(0) != target.size(0):
            raise IndexError("Batch size mismatch between preds and target.")

        target = target.view(-1, 1)
        abs_diff = torch.abs(target - self.midpoints.to(preds.device).unsqueeze(0))
        
        cross_entropy = torch.sum((abs_diff ** self.p) * preds, dim=1)
        shannon_entropy = torch.sum(preds * torch.log(preds.clamp_min(1e-10)), dim=1)

        losses = cross_entropy - self.tau * shannon_entropy
        loss = losses.sum()

        return loss
