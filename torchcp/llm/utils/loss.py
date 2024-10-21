
import torch
def set_losses_from_labels(set_labels):
    """Given individual labels, compute set loss."""
    return torch.cumprod(1 - set_labels, axis=-1)