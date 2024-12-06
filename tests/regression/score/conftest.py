import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, TensorDataset


@pytest.fixture
def dummy_data():
    """
    Fixture to provide dummy data for testing.
    """
    x_train = torch.rand((100, 10))
    y_train = torch.rand((100, 1))
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16)

    x_test = torch.rand((20, 10))
    y_test = torch.rand((20, 1))
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    return train_dataloader, test_dataloader
