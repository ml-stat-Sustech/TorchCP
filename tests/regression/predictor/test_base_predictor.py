import pytest
from torchcp.regression.score import ABS
from torchcp.regression.predictor.base import BasePredictor

import pytest
import torch
from torch.utils.data import DataLoader

@pytest.fixture
def mock_score_function():
    return ABS()

# Partial implementation of BasePredictor for testing
class PartialPredictor(BasePredictor):
    def __init__(self, score_function, model=None):
        super().__init__(score_function, model)
        # Intentionally not implementing any abstract methods

@pytest.mark.parametrize("method_name", [
    "fit",
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
        "fit": (train_dataloader,),
        "calculate_score": (torch.rand(10, 5), torch.rand(10,)),
        "generate_intervals": (torch.rand(10, 5), 0.5),
        "predict": (torch.rand(10, 5),),
        "calibrate": (cal_dataloader, 0.1),
        "evaluate": (test_dataloader,),
    }

    with pytest.raises(NotImplementedError):
        # Dynamically call the method
        getattr(partial_predictor, method_name)(*dummy_args[method_name])

