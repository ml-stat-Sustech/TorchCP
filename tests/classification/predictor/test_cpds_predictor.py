# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import pytest

from torchcp.regression.predictor import ConformalPredictiveDistribution
from torchcp.regression.score import ABS
from torchcp.regression.utils import build_regression_model
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def mock_model():
    return build_regression_model("NonLinearNet")(5, 1, 64, 0.5)


@pytest.fixture
def cpds_predictor(mock_model):
    return ConformalPredictiveDistribution(model=mock_model)


def test_invalid_initialization():
    with pytest.raises(ValueError):
        ConformalPredictiveDistribution(score_function=ABS())

@pytest.fixture
def mock_data():
    """
    提供用于回归预测器测试的假数据。
    返回: (train_loader, cal_loader, test_loader)
    - 特征维度为 5，目标为连续值
    """
    num_samples = 3000
    num_features = 5

    # 生成特征
    X = torch.rand((num_samples, num_features), dtype=torch.float32)

    # 生成一个平滑的连续目标（非线性 + 噪声）
    y_wo_noise = (
        10 * torch.sin(X[:, 0] * X[:, 1] * torch.pi)
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    noise = 0.5 * torch.randn(num_samples, dtype=torch.float32)
    y = (y_wo_noise + noise).to(torch.float32)

    # 划分 train/cal/test（40%/20%/40%）
    indices = torch.randperm(num_samples)
    split_index1 = int(num_samples * 0.4)
    split_index2 = int(num_samples * 0.6)
    part1 = indices[:split_index1]
    part2 = indices[split_index1:split_index2]
    part3 = indices[split_index2:]

    train_dataset = TensorDataset(X[part1], y[part1])
    cal_dataset = TensorDataset(X[part2], y[part2])
    test_dataset = TensorDataset(X[part3], y[part3])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    cal_loader = DataLoader(cal_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, cal_loader, test_loader

def test_workflow(mock_data, cpds_predictor):
    # Extract mock data
    _, cal_dataloader, test_dataloader = mock_data

    cpds_predictor.calibrate(cal_dataloader)
    assert hasattr(cpds_predictor, "scores"), "SplitPredictor should have scores after calibration."

    for x_batch, _ in test_dataloader:
            cpds_predictor.predict(x_batch)

    with pytest.raises(ValueError):
        tmp_cpds_predictor = ConformalPredictiveDistribution()
        for x_batch, _ in test_dataloader:
            tmp_cpds_predictor.predict(x_batch)