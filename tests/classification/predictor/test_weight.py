import pytest
import torch
from torch.utils.data import Dataset

from torchcp.classification.scores import THR
from torchcp.classification.predictors import WeightedPredictor


@pytest.fixture
def mock_dataset():
    class MyDataset(Dataset):
        def __init__(self):
            self.x = torch.randn(100, 3)
            self.labels = torch.randint(0, 3, (100, ))

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.labels[idx]
    return MyDataset()


@pytest.fixture
def mock_score_function():
    return THR(score_type="softmax")


@pytest.fixture
def mock_model():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x
    return MockModel()


@pytest.fixture
def mock_image_encoder():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x
    return MockModel()


@pytest.fixture
def predictor(mock_score_function, mock_model, mock_image_encoder):
    return WeightedPredictor(mock_score_function, mock_model, 2.0, mock_image_encoder)


def test_valid_initialization(predictor, mock_score_function, mock_model, mock_image_encoder):
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model
    assert not predictor._model.training
    assert predictor.image_encoder is mock_image_encoder
    assert predictor._device == next(mock_model.parameters()).device
    assert predictor._logits_transformation.temperature == 2.0
    assert predictor.scores is None
    assert predictor.alpha is None


def test_invalid_initialization(mock_score_function, mock_model):
    with pytest.raises(ValueError, match="image_encoder cannot be None."):
        WeightedPredictor(mock_score_function, mock_model)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calibrate(predictor, mock_dataset, mock_score_function, mock_model, alpha):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha)

    assert torch.equal(predictor.source_image_features, mock_dataset.x)
    
    mock_model.eval()
    logits = mock_model(mock_dataset.x) / 2.0
    labels = mock_dataset.labels

    scores = torch.zeros(logits.shape[0] + 1)
    scores[:logits.shape[0]] = mock_score_function(logits, labels)
    scores[logits.shape[0]] = torch.tensor(torch.inf)
    assert torch.equal(predictor.scores, scores)
    assert torch.equal(predictor.scores_sorted, scores.sort()[0])

@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calculate_threshold(predictor, mock_score_function, alpha):
    logits = torch.randn(100, 3)
    labels = torch.randint(0, 3, (100, ))

    predictor.calculate_threshold(logits, labels, alpha)
    assert predictor.alpha == alpha

    scores = torch.zeros(logits.shape[0] + 1)
    scores[:logits.shape[0]] = mock_score_function(logits, labels)
    scores[logits.shape[0]] = torch.tensor(torch.inf)
    assert torch.equal(predictor.scores, scores)
    assert torch.equal(predictor.scores_sorted, scores.sort()[0])


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 2])
def test_invalid_calculate_threshold(predictor, alpha):
    logits = torch.randn(100, 3)
    labels = torch.randint(0, 3, (100, ))
    with pytest.raises(ValueError, match="alpha should be a value"):
        predictor.calculate_threshold(logits, labels, alpha)
    

# @pytest.mark.parametrize("q_hat", [0.1, 0.05])
# def test_predict(predictor, mock_score_function, mock_model, mock_dataset, q_hat):
#     predictor.q_hat = q_hat
#     pred_sets = predictor.predict(mock_dataset.x)

#     logits = mock_model(mock_dataset.x)
#     scores = mock_score_function(logits)
#     excepted_sets = [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
#     assert pred_sets == excepted_sets


# @pytest.mark.parametrize("q_hat", [0.5, 0.7])
# def test_evaluate(predictor, mock_score_function, mock_model, mock_dataset, q_hat):

#     cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
#     predictor.q_hat = q_hat
#     results = predictor.evaluate(cal_dataloader)

#     logits = mock_model(mock_dataset.x)
#     scores = mock_score_function(logits)
#     excepted_sets = [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
#     metrics = Metrics()
#     assert len(results) == 2
#     assert results['Coverage_rate'] == metrics('coverage_rate')(excepted_sets, mock_dataset.labels)
#     assert results['Average_size'] == metrics('average_size')(excepted_sets, mock_dataset.labels)
