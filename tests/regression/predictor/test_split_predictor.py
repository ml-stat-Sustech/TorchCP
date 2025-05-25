# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import pytest

from torchcp.regression.predictor import SplitPredictor
from torchcp.regression.score import ABS
from torchcp.regression.utils import build_regression_model


@pytest.fixture
def mock_model():
    return build_regression_model("NonLinearNet")(5, 1, 64, 0.5)


@pytest.fixture
def mock_score_function():
    return ABS()


@pytest.fixture
def split_predictor(mock_model, mock_score_function):
    return SplitPredictor(score_function=mock_score_function, model=mock_model)


@pytest.fixture
def split_predictor_nomodel(mock_score_function):
    return SplitPredictor(score_function=mock_score_function)


def test_initialization_device(mock_score_function):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = SplitPredictor(mock_score_function, device=device)
    assert predictor._device == device


def test_workflow(mock_data, split_predictor, split_predictor_nomodel, mock_model):
    # Extract mock data
    train_dataloader, cal_dataloader, test_dataloader = mock_data

    # Step 1: train the model
    # case 1
    split_predictor_nomodel.train(train_dataloader)
    for param in split_predictor_nomodel._model.parameters():
        assert param.grad is not None, "Model parameters should have gradients."
    # case 2
    split_predictor_nomodel.train(train_dataloader, model=mock_model)
    for param in split_predictor_nomodel._model.parameters():
        assert param.grad is not None, "Model parameters should have gradients."
    # case 3
    split_predictor.train(train_dataloader)
    for param in split_predictor._model.parameters():
        assert param.grad is not None, "Model parameters should have gradients."

    # Step 2: Calibrate the model
    alpha = 0.1
    split_predictor.calibrate(cal_dataloader, alpha=alpha)
    assert hasattr(split_predictor, "scores"), "SplitPredictor should have scores after calibration."
    assert hasattr(split_predictor, "q_hat"), "SplitPredictor should have q_hat after calibration."

    # Step 3: Evaluate the model
    eval_res = split_predictor.evaluate(test_dataloader)
    assert "coverage_rate" in eval_res, "Coverage rate should be part of evaluation results."
    assert "average_size" in eval_res, "Average size should be part of evaluation results."
    assert abs(
        eval_res['coverage_rate'] - 0.9) < 5e-2, f"Coverage rate {eval_res['coverage_rate']} should be close to 0.9"
    assert eval_res["average_size"] > 0, "Average size should be greater than 0."
