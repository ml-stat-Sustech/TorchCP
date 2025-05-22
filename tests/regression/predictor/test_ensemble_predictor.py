# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import torch

from torchcp.regression.predictor import EnsemblePredictor
from torchcp.regression.score import ABS
from torchcp.regression.utils import build_regression_model


@pytest.fixture
def mock_model():
    return build_regression_model("NonLinearNet")(5, 1, 64, 0.5)


@pytest.fixture
def mock_score_function():
    return ABS()


@pytest.mark.parametrize("aggregation_function", ['mean', 'median', lambda x, dim: torch.max(x, dim=dim)[0]])
def test_workflow(mock_data, mock_model, mock_score_function, aggregation_function):
    train_dataloader, cal_dataloader, test_dataloader = mock_data
    
    # initialize EnsemblePredictor
    ensemble_predictor = EnsemblePredictor(
        score_function=mock_score_function,
        model=mock_model,
        aggregation_function=aggregation_function
    )

    # test train method
    ensemble_predictor.train(train_dataloader, ensemble_num=3, subset_num=100)
    assert len(ensemble_predictor.model_list) == 3, "Ensemble should contain 3 models."
    assert len(ensemble_predictor.indices_list) == 3, "Each model should have corresponding indices."

    # test predict method
    x_batch, y_batch = next(iter(test_dataloader))
    prediction_intervals, aggr_pred = ensemble_predictor.predict(alpha=0.1, x_batch=x_batch)
    assert prediction_intervals is not None, "Prediction intervals should be generated."
    assert aggr_pred is not None, "Aggregated predictions should be generated."
    assert prediction_intervals.shape[0] == x_batch.shape[0], "Prediction intervals should match batch size."

    # test calibrate method
    ensemble_predictor.calibrate(cal_dataloader, alpha=0.1)

    # test evaluate method
    eval_res = ensemble_predictor.evaluate(test_dataloader, alpha=0.1, verbose=False)
    # Test evaluate method with verbose=True
    ensemble_predictor.evaluate(test_dataloader, alpha=0.1, verbose=True)

    assert "total batches" in eval_res, "Evaluation should return total batches."
    assert eval_res["total batches"] > 0, "There should be at least one batch evaluated."
    assert eval_res["coverage_rate"] > 0, "Coverage rate should be greater than 0."
    assert eval_res["average_size"] > 0, "Interval size should be greater than 0."

    # Test evaluate method with verbose=True
    ensemble_predictor.evaluate(test_dataloader, alpha=0.1, verbose=True)

    with pytest.raises(ValueError):
        ensemble_predictor.train(train_dataloader, ensemble_num=0, subset_num=100)

    with pytest.raises(ValueError):
        ensemble_predictor.predict(alpha=0.1, x_batch=x_batch, y_batch_last=torch.rand(10), aggr_pred_last=None)

def test_wrong_workflow(mock_data, mock_model, mock_score_function):
    with pytest.raises(ValueError):
        EnsemblePredictor(mock_score_function, mock_model, aggregation_function='error')

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_support(mock_data, mock_model, mock_score_function, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    train_dataloader, cal_dataloader, test_dataloader = mock_data
    ensemble_predictor = EnsemblePredictor(mock_score_function, mock_model.to(device))

    ensemble_predictor.train(train_dataloader, ensemble_num=3, subset_num=100)
    x_batch, _ = next(iter(test_dataloader))
    x_batch = x_batch.to(device)

    prediction_intervals, _ = ensemble_predictor.predict(alpha=0.1, x_batch=x_batch)
    assert prediction_intervals.device.type == device, "Device should match the specified device."
