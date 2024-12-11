# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.regression.score.base import BaseScore


class IncompleteScore(BaseScore):
    def __init__(self) -> None:
        super().__init__()


@pytest.mark.parametrize("method_name", [
    "__call__",
    "construct_interval",
    "train",
])
def test_not_implemented_methods(method_name, dummy_data):
    """
    Test that calling any unimplemented method in IncompleteScore raises NotImplementedError.
    """
    incomplete_score = IncompleteScore()
    train_dataloader, _ = dummy_data

    # Define dummy arguments for each method
    dummy_args = {
        "__call__": (torch.rand(10, 5), torch.rand(10, )),
        "construct_interval": (torch.rand(10, 5), 0.5),
        "train": (None, 10, train_dataloader, None, None),  # None values for model and criterion
    }

    with pytest.raises(NotImplementedError):
        # Dynamically call the method
        getattr(incomplete_score, method_name)(*dummy_args[method_name])
