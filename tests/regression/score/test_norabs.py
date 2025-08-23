import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchcp.regression.score.abs import ABS
from torchcp.regression.score.norabs import NorABS, DifficultyEstimator, TorchMinMaxScaler

import torch
from torch.utils.data import TensorDataset, DataLoader

from torchcp.regression.predictor import SplitPredictor
from torchcp.regression.utils import build_regression_model
from torchcp.regression.score.norabs import NorABS


def test_splitpredictor_with_norabs():
    X = torch.randn(40, 3)
    y = torch.randn(40)
    train_dataset = TensorDataset(X[:20], y[:20])
    cal_dataset = TensorDataset(X[20:30], y[20:30])
    test_dataset = TensorDataset(X[30:], y[30:])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    cal_loader = DataLoader(cal_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_regression_model("GaussianRegressionModel")(X.shape[1], 16, 0.1)

    score_function = NorABS(data_loader=train_loader, estimate_type="variance", k=5, scalar=True, beta=0.01, device=device)
    predictor = SplitPredictor(score_function=score_function, model=model, alpha=0.1, device=device)

    predictor.train(train_loader, epochs=3, lr=0.01, device=device, verbose=True)

    predictor.calibrate(cal_loader)
    
    for i in tqdm(range(len(test_dataset)), desc="Online Evaluation"):
        x_test, y_test = test_dataset[i]
        x_test = x_test.to(device)
        pred_interval = predictor.predict(x_test.unsqueeze(0))[0][0]

        assert pred_interval.shape == (2,), "Prediction interval should be of shape (2,)"
        assert pred_interval[0] <= pred_interval[1], "Lower bound should not exceed upper bound"
        break 

# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def dummy_data():
    X = torch.randn(50, 4)
    y = torch.randn(50)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)
    return loader


@pytest.fixture
def simple_model():
    """A trivial regression model for calibration tests."""
    class SimpleModel(nn.Module):
        def __init__(self, in_dim=4):
            super().__init__()
            self.fc = nn.Linear(in_dim, 2)  # mean + variance

        def forward(self, x):
            out = self.fc(x)
            return torch.cat([out[:, :1], torch.abs(out[:, 1:]) + 1e-6], dim=1)

    return SimpleModel()


# -------------------------------
# TorchMinMaxScaler Tests
# -------------------------------
def test_minmax_scaler_fit_transform_and_clip():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    scaler = TorchMinMaxScaler(clip=True)
    out = scaler.fit_transform(data)
    assert torch.all((out >= 0) & (out <= 1)), "Scaler should produce values in [0, 1]"


def test_minmax_scaler_no_fit_raises():
    scaler = TorchMinMaxScaler()
    with pytest.raises(RuntimeError):
        _ = scaler.transform(torch.randn(2, 2))


def test_minmax_scaler_constant_feature():
    data = torch.ones(5, 2)
    scaler = TorchMinMaxScaler().fit(data)
    out = scaler.transform(data)
    assert torch.allclose(out, torch.zeros_like(out)), "Constant feature should normalize to 0"


# -------------------------------
# DifficultyEstimator Tests
# -------------------------------
def test_invalid_estimator_type(dummy_data):
    with pytest.raises(ValueError):
        DifficultyEstimator(dummy_data, estimator_type="invalid")


def test_function_mode_without_callable(dummy_data):
    with pytest.raises(ValueError):
        DifficultyEstimator(dummy_data, estimator_type="function")


def test_variance_mode_with_invalid_predicts(dummy_data, simple_model):
    est = DifficultyEstimator(dummy_data, estimator_type="variance")
    est.calibrate(simple_model)
    with pytest.raises(ValueError):
        est._compute_difficulty(torch.randn(2, 2), torch.randn(2, 3))


def test_function_mode_with_invalid_inputs(dummy_data, simple_model):
    est = DifficultyEstimator(dummy_data, estimator_type="function", custom_function=lambda x, y: y[:, 1])
    est.calibrate(simple_model)
    with pytest.raises(ValueError):
        est._compute_difficulty(None, None)


def test_knn_modes(dummy_data, simple_model):
    for mode in ["knn_distance", "knn_label", "knn_residual"]:
        est = DifficultyEstimator(dummy_data, estimator_type=mode, k=3)
        est.calibrate(simple_model)
        scores = est.apply(torch.randn(2, 4))
        assert scores.shape[0] == 2


def test_apply_without_calibrate(dummy_data):
    est = DifficultyEstimator(dummy_data, estimator_type="variance")
    with pytest.raises(RuntimeError):
        est.apply(torch.randn(2, 4), torch.randn(2, 2))


# -------------------------------
# NorABS Tests
# -------------------------------
def test_norabs_call_and_generate_intervals(dummy_data, simple_model):
    norabs = NorABS(dummy_data, estimate_type="variance")
    norabs.calibrate(simple_model)

def test_norabs_call_and_calibration(dummy_data, simple_model):
    norabs = NorABS(dummy_data, estimate_type="variance")
    model = simple_model
    x = torch.randn(2, 1)
    predicts = torch.randn(2, 2)
    y_truth = torch.randn(2)
    scores = norabs(predicts, y_truth, x_batch=x, model=model)
    assert scores.shape == (2, 1)


def test_norabs_call_without_model_raises(dummy_data):
    norabs = NorABS(dummy_data, estimate_type="variance")
    with pytest.raises(ValueError):
        norabs(torch.randn(2, 2), torch.randn(2))


def test_norabs_call_without_model(dummy_data):
    norabs = NorABS(dummy_data, estimate_type="variance")
    predicts = torch.randn(3, 2)
    y_truth = torch.randn(3)
    with pytest.raises(ValueError):
        _ = norabs(predicts, y_truth)


def test_norabs_train_returns_model(dummy_data):
    norabs = NorABS(dummy_data, estimate_type="variance")
    model = norabs.train(dummy_data, epochs=1, verbose=False)
    sample_input, _ = next(iter(dummy_data))
    out = model(sample_input)
    assert out.shape[1] == 2
