import pytest
import torch

from torchcp.regression.predictor import ACIPredictor
from torchcp.regression.score import CQR
from torchcp.regression.utils import build_regression_model


@pytest.fixture
def mock_model():
    return build_regression_model("NonLinearNet")(5, 2, 64, 0.5)


@pytest.fixture
def mock_score_function():
    return CQR()


def test_aci_predictor_workflow(mock_data, mock_model, mock_score_function):
    train_dataloader, cal_dataloader, test_dataloader = mock_data

    with pytest.raises(ValueError, match="gamma must be greater than 0."):
        ACIPredictor(mock_model, mock_score_function, gamma=0)

    # Initialize ACIPredictor
    aci_predictor = ACIPredictor(
        model=mock_model,
        score_function=mock_score_function,
        gamma=0.1
    )

    # Test train method
    alpha = 0.1
    aci_predictor.train(train_dataloader, alpha=alpha)
    assert aci_predictor.alpha == alpha, "Alpha should be set correctly during training."
    assert aci_predictor.alpha_t == alpha, "Adaptive alpha_t should start with initial alpha."

    # Test predict method
    x_batch, y_batch = next(iter(test_dataloader))
    prediction_intervals = aci_predictor.predict(x_batch)
    assert prediction_intervals is not None, "Prediction intervals should not be None."
    assert prediction_intervals.shape[0] == x_batch.shape[0], "Prediction intervals should match batch size."

    # test calibrate method
    aci_predictor.calibrate(cal_dataloader, alpha=0.1)

    # Test evaluate method
    eval_results = aci_predictor.evaluate(test_dataloader, verbose=False)
    assert eval_results["Total batches"] > 0, "Evaluation should process at least one batch."
    assert eval_results["Coverage_rate"] > 0, "Average coverage rate should be greater than 0."
    assert eval_results["Average_size"] > 0, "Average interval size should be greater than 0."

    # Test evaluate method with verbose=True
    aci_predictor.evaluate(test_dataloader, verbose=True)

    # Test exception for missing arguments
    with pytest.raises(ValueError):
        aci_predictor.predict(x_batch, y_batch_last=torch.rand(10), pred_interval_last=None)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_support(mock_data, mock_model, mock_score_function, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    train_dataloader, _, test_dataloader = mock_data
    aci_predictor = ACIPredictor(mock_model.to(device), mock_score_function, gamma=0.1)

    aci_predictor.train(train_dataloader, alpha=0.1)
    x_batch, _ = next(iter(test_dataloader))
    x_batch = x_batch.to(device)

    prediction_intervals = aci_predictor.predict(x_batch)
    assert prediction_intervals.device.type == device, "Prediction intervals should match the specified device."
