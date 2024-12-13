import pytest
import torch

from torchcp.regression.score.base import BaseScore


class IncompleteScore(BaseScore):
    def __init__(self) -> None:
        super().__init__()


@pytest.mark.parametrize("method_name", [
    "__call__",
    "generate_intervals",
    "train",
])
def test_not_implemented_methods(method_name, dummy_data):
    """
    Test that calling any unimplemented method in IncompleteScore raises NotImplementedError.
    """
    incomplete_score = IncompleteScore()
    train_dataloader, _ = dummy_data

    dummy_args = {
        "__call__": (torch.rand(10, 5), torch.rand(10, )),
        "generate_intervals": (torch.rand(10, 5), 0.5),
        "train": (None, 10, train_dataloader, None, None), 
    }

    with pytest.raises(NotImplementedError):
        getattr(incomplete_score, method_name)(*dummy_args[method_name])
