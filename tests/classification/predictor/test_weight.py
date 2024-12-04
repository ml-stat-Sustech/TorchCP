import pytest
import torch
from torch.utils.data import Dataset

from torchcp.classification.score import THR
from torchcp.classification.predictor import WeightedPredictor
from torchcp.classification.predictor.utils import build_DomainDetecor


class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.randn(100, 3)
        self.labels = torch.randint(0, 3, (100, ))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]
    

@pytest.fixture
def mock_dataset():
    return MyDataset()


@pytest.fixture
def mock_val_dataset():
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
def mock_domain_classifier():
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x[:, :2]
    return MockModel()


@pytest.fixture
def predictor(mock_score_function, mock_model, mock_image_encoder, mock_domain_classifier):
    return WeightedPredictor(mock_score_function, mock_model, 1.0, mock_image_encoder, mock_domain_classifier)


def test_valid_initialization(predictor, mock_score_function, mock_model, mock_image_encoder):
    assert predictor.score_function is mock_score_function
    assert predictor._model is mock_model
    assert not predictor._model.training
    assert predictor.image_encoder is mock_image_encoder
    assert predictor._device == next(mock_model.parameters()).device
    assert predictor._logits_transformation.temperature == 1.0
    assert predictor.scores is None
    assert predictor.alpha is None


def test_invalid_initialization(mock_score_function, mock_model):
    with pytest.raises(ValueError, match="image_encoder cannot be None."):
        WeightedPredictor(mock_score_function, mock_model, domain_classifier=mock_model)


@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_calibrate(predictor, mock_dataset, mock_score_function, mock_model, alpha):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha)

    assert torch.equal(predictor.source_image_features, mock_dataset.x)
    
    mock_model.eval()
    logits = mock_model(mock_dataset.x) / 1.0
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
    

@pytest.mark.parametrize("alpha", [0.1, 0.05])
def test_predict(predictor, mock_dataset, mock_val_dataset, alpha):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha)
    pred_sets = predictor.predict(mock_val_dataset.x)

    w_new = mock_val_dataset.x[:, 1] / mock_val_dataset.x[:, 0]
    w_sorted = (mock_dataset.x[:, 1] / mock_dataset.x[:, 0]).sort(descending=False)[0].expand([100, -1])
    w_sorted = torch.cat([w_sorted, w_new.unsqueeze(1)], 1)
    p_sorted = w_sorted / w_sorted.sum(1, keepdim=True)
    p_sorted_acc = p_sorted.cumsum(1)
    i_T = torch.argmax((p_sorted_acc >= 1.0 - alpha).int(), dim=1, keepdim=True)
    q_hat_batch = predictor.scores_sorted.expand([100, -1]).gather(1, i_T).detach()

    logits = mock_val_dataset.x
    excepted_sets = []
    for _, (logits_instance, q_hat) in enumerate(zip(logits, q_hat_batch)):
        excepted_sets.extend(predictor.predict_with_logits(logits_instance, q_hat))
    assert pred_sets == excepted_sets


def test_invalid_predict(predictor, mock_dataset, mock_val_dataset, mock_score_function, mock_model, mock_image_encoder):
    with pytest.raises(ValueError, match="Please calibrate first to get self.scores_sorted"):
        predictor.predict(mock_val_dataset.x)
    
    predictor = WeightedPredictor(mock_score_function, mock_model, 1.0, mock_image_encoder, None)
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, 0.1)
    pred_sets = predictor.predict(mock_val_dataset.x)
    assert len(pred_sets) == 100


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_evaluate(predictor, mock_dataset, mock_val_dataset, alpha):
    val_dataloader = torch.utils.data.DataLoader(mock_val_dataset, batch_size=40)
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, alpha)
    results = predictor.evaluate(val_dataloader)
    assert len(results) == 2
    assert "Coverage_rate" in results
    assert "Average_size" in results


def test_invalid_evaluate(predictor, mock_dataset, mock_val_dataset, mock_score_function, mock_model, mock_image_encoder):
    val_dataloader = torch.utils.data.DataLoader(mock_val_dataset, batch_size=40)
    with pytest.raises(ValueError, match="Please calibrate first to get self.source_image_features"):
        predictor.evaluate(val_dataloader)
    
    predictor = WeightedPredictor(mock_score_function, mock_model, 1.0, mock_image_encoder, None)
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, 0.1)
    results = predictor.evaluate(val_dataloader)

    assert len(results) == 2
    assert "Coverage_rate" in results
    assert "Average_size" in results


def test_train_domain_classifier(predictor, mock_dataset, mock_val_dataset):
    cal_dataloader = torch.utils.data.DataLoader(mock_dataset, batch_size=40)
    predictor.calibrate(cal_dataloader, 0.1)

    target_image_features = mock_val_dataset.x
    predictor._train_domain_classifier(target_image_features)

    def models_are_equal(model1, model2):
        if len(list(model1.children())) != len(list(model2.children())):
            return False
        for (layer1, layer2) in zip(model1.children(), model2.children()):
            if type(layer1) != type(layer2):
                return False
        return True
    domain_classifier = build_DomainDetecor(target_image_features.shape[1], 2, 'cpu')
    assert models_are_equal(predictor.domain_classifier, domain_classifier)
    assert next(predictor.domain_classifier.parameters()).device == next(domain_classifier.parameters()).device