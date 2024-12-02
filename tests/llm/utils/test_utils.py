import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchcp.llm.utils.utils import calculate_midpoints

# FILE: torchcp/regression/utils/test_utils.py


@pytest.fixture
def data_loader():
    data = [torch.tensor([i, i + 1]) for i in range(10)]
    dataset = TensorDataset(torch.arange(len(data)), torch.stack(data))
    return DataLoader(dataset, batch_size=2)

def test_calculate_midpoints(data_loader):
    K = 5
    midpoints = calculate_midpoints(data_loader, K)
    
    assert isinstance(midpoints, torch.Tensor)
    assert midpoints.shape[0] == K
    assert torch.all(midpoints >= 0)
    assert torch.all(midpoints <= 10)

def test_calculate_midpoints_empty():
    data_loader = DataLoader(TensorDataset(torch.tensor([]), torch.tensor([])), batch_size=2)
    K = 5
    with pytest.raises(RuntimeError):
        calculate_midpoints(data_loader, K)