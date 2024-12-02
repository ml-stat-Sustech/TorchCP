import pytest
import torch
from torchcp.regression.scores import split
from torchcp.regression.utils import build_regression_model
from torchcp.regression.predictors import EnsemblePredictor

@pytest.fixture
def mock_model():
    return build_regression_model("NonLinearNet")(5, 1, 64, 0.5)

@pytest.fixture
def mock_score_function():
    return split()

@pytest.fixture
def ensemble_predictor(mock_model, mock_score_function):
    return EnsemblePredictor(model=mock_model, score_function=mock_score_function, aggregation_function='mean')

def test_workflow(mock_data, ensemble_predictor):
    # Extract mock data
    train_dataloader, cal_dataloader, test_dataloader = mock_data

    # Step 1: Fit the ensemble model
    ensemble_predictor.fit(train_dataloader, ensemble_num=5, subset_num=500)
    for model in ensemble_predictor.model_list:
        for param in model.parameters():
            assert param.grad is not None, "Model parameters should have gradients."

    # Step 2: Calibrate the model
    alpha = 0.1
    ensemble_predictor.calibrate(cal_dataloader, alpha=alpha)
    assert hasattr(ensemble_predictor, "scores"), "EnsemblePredictor should have scores after calibration."
    assert hasattr(ensemble_predictor, "q_hat"), "EnsemblePredictor should have q_hat after calibration."

    # Step 3: Evaluate the ensemble model
    eval_res = ensemble_predictor.evaluate(test_dataloader, alpha=alpha)
    assert "Total batches" in eval_res, "Total batches should be part of evaluation results."
    assert "Average coverage rate" in eval_res, "Average coverage rate should be part of evaluation results."
    assert "Average prediction interval size" in eval_res, "Average size should be part of evaluation results."
    assert torch.isclose(torch.tensor(eval_res['Average coverage rate']), torch.tensor(0.9), atol=5e-1), "Coverage rate should be close to 0.9."
    assert eval_res["Average prediction interval size"] > 0, "Average size should be greater than 0."
    
    # Step 4: Predict with the ensemble model
    x_batch, y_batch = next(iter(test_dataloader))
    prediction_intervals, aggr_pred = ensemble_predictor.predict(alpha=alpha, x_batch=x_batch, y_batch_last=None, aggr_pred_last=None)
    
    # Ensure prediction intervals are returned
    assert prediction_intervals is not None, "Prediction intervals should be generated."
    assert aggr_pred is not None, "Aggregated predictions should be generated."
    assert prediction_intervals.shape[0] == x_batch.shape[0], "Prediction intervals should match the batch size."
