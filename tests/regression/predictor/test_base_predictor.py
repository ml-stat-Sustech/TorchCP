# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.regression.predictor.base import BasePredictor
from torchcp.regression.score import ABS


@pytest.fixture
def mock_score_function():
    return ABS()


# Partial implementation of BasePredictor for testing
class PartialPredictor(BasePredictor):
    def __init__(self, score_function, model=None, alpha=0.1):
        super().__init__(score_function, model, alpha)
        # Intentionally not implementing any abstract methods


@pytest.mark.parametrize("method_name", [
    "train",
    "calculate_score",
    "generate_intervals",
    "predict",
    "evaluate",
    "calibrate",
])
def test_not_implemented_methods(method_name, mock_data, mock_score_function):
    """
    Test that calling any unimplemented method in PartialPredictor raises NotImplementedError.
    """
    partial_predictor = PartialPredictor(score_function=mock_score_function)
    train_dataloader, cal_dataloader, test_dataloader = mock_data

    # Define dummy arguments for each method
    dummy_args = {
        "train": (train_dataloader,),
        "calculate_score": (torch.rand(10, 5), torch.rand(10, )),
        "generate_intervals": (torch.rand(10, 5), 0.5),
        "predict": (torch.rand(10, 5),),
        "calibrate": (cal_dataloader, 0.1),
        "evaluate": (test_dataloader,),
    }

    with pytest.raises(NotImplementedError):
        # Dynamically call the method
        getattr(partial_predictor, method_name)(*dummy_args[method_name])


@pytest.mark.parametrize("alpha", [-1, 0, 1, 2])
def test_initializatioin_alpha(mock_score_function, alpha):
    with pytest.raises(ValueError, match="alpha should be a value"):
        PartialPredictor(score_function=mock_score_function, alpha=alpha)


def test_model_type_error(mock_score_function):
    """
    Test that a TypeError is raised when model is not an instance of torch.nn.Module.
    """
    with pytest.raises(TypeError, match="The model must be an instance of torch.nn.Module"):
        PartialPredictor(score_function=mock_score_function, model="not_a_model")
