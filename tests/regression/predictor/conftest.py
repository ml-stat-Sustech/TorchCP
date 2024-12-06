import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


@pytest.fixture
def mock_data():
    n = 10000
    X = np.random.rand(n, 5)
    y_wo_noise = 10 * np.sin(X[:, 0] * X[:, 1] * np.pi) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    eplison = np.zeros(n)
    phi = theta = 0.8
    delta_t_1 = np.random.randn()
    for i in range(1, n):
        delta_t = np.random.randn()
        eplison[i] = phi * eplison[i - 1] + delta_t_1 + theta * delta_t
        delta_t_1 = delta_t

    y = y_wo_noise + eplison

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index1 = int(len(indices) * 0.4)
    split_index2 = int(len(indices) * 0.6)
    part1, part2, part3 = np.split(indices, [split_index1, split_index2])
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[part1, :])
    train_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part1, :])), torch.from_numpy(y[part1]))
    cal_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part2, :])), torch.from_numpy(y[part2]))
    test_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part3, :])), torch.from_numpy(y[part3]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)
    cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

    return train_dataloader, cal_dataloader, test_dataloader
