import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchcp.classification.utils.conf_calibration import ConfCalibrator, Identity, TS, oTS, optimze_oTS, ConfCalibrator_REGISTRY, ConfOptimizer_REGISTRY

@pytest.fixture
def dataloader():
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=10)

def test_identity():
    model = Identity()
    logits = torch.randn(10, 5)
    output = model(logits)
    assert torch.equal(output, logits)

def test_ts():
    model = TS(temperature=2.0)
    logits = torch.randn(10, 5)
    output = model(logits)
    assert torch.equal(output, logits / 2.0)

def test_ots():
    model = oTS(temperature=2.0)
    logits = torch.randn(10, 5)
    output = model(logits)
    assert torch.equal(output, logits / 2.0)

def test_optimze_oTS(dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformation = oTS(temperature=1.0)
    optimized_transformation = optimze_oTS(transformation, dataloader, device, max_iters=100, lr=0.01, epsilon=0.01)
    assert isinstance(optimized_transformation, oTS)
    assert optimized_transformation.temperature.item() != 1.0

def test_registry_conf_calibrator():
    with pytest.raises(NameError):
        ConfCalibrator.registry_ConfCalibrator("undefined_calibrator")

    calibrator = ConfCalibrator.registry_ConfCalibrator("Identity")
    assert isinstance(calibrator(), Identity)

def test_registry_conf_optimizer():
    with pytest.raises(NameError):
        ConfCalibrator.registry_ConfOptimizer("undefined_optimizer")

    optimizer = ConfCalibrator.registry_ConfOptimizer("optimze_oTS")
    assert optimizer == optimze_oTS

if __name__ == '__main__':
    pytest.main()