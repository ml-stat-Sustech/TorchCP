# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import pytest

from torchcp.regression.predictor import ConformalPredictiveDistribution
from torchcp.regression.score import ABS
from torchcp.regression.utils import build_regression_model


@pytest.fixture
def mock_model():
    return build_regression_model("NonLinearNet")(5, 1, 64, 0.5)


@pytest.fixture
def cpds_predictor(mock_model):
    return ConformalPredictiveDistribution(model=mock_model)


def test_invalid_initialization():
    with pytest.raises(ValueError):
        ConformalPredictiveDistribution(score_function=ABS())


def test_workflow(mock_data, cpds_predictor):
    # Extract mock data
    _, cal_dataloader, test_dataloader = mock_data

    cpds_predictor.calibrate(cal_dataloader)
    assert hasattr(cpds_predictor, "scores"), "SplitPredictor should have scores after calibration."

    for x_batch, _ in test_dataloader:
            cpds_predictor.predict(x_batch)

    with pytest.raises(ValueError):
        tmp_cpds_predictor = ConformalPredictiveDistribution()
        for x_batch, _ in test_dataloader:
            tmp_cpds_predictor.predict(x_batch)