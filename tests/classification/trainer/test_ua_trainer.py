# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import pytest
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchcp.classification.loss import UncertaintyAwareLoss
from torchcp.classification.trainer import UncertaintyAwareTrainer


class ClassifierDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return len(self.X_data)


class TrainDataset(Dataset):
    def __init__(self, X_data, Y_data, Z_data):
        self.X_data = X_data
        self.Y_data = Y_data
        self.Z_data = Z_data

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index], self.Z_data[index]

    def __len__(self):
        return len(self.X_data)


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.param * x


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_conflearn_trainer(mock_model):
    optimizer = optim.Adam(mock_model.parameters(), lr=0.1)
    return UncertaintyAwareTrainer(mock_model, optimizer, device='cpu')


@pytest.fixture
def train_loader():
    X_train = torch.rand((100, 3)).float()
    Y_train = torch.randint(0, 3, (100,)).long()
    train_dataset = ClassifierDataset(X_train, Y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=20, shuffle=True, drop_last=True)
    return train_loader


@pytest.fixture
def val_loader():
    X_val = torch.rand((100, 3)).float()
    Y_val = torch.randint(0, 3, (100,)).long()
    val_dataset = ClassifierDataset(X_val, Y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=20, shuffle=True, drop_last=True)
    return val_loader


def test_initialization(mock_model):
    optimizer = optim.Adam(mock_model.parameters(), lr=0.1)
    conflearn_trainer = UncertaintyAwareTrainer(mock_model, optimizer, device='cpu')
    assert conflearn_trainer.model is mock_model
    assert conflearn_trainer.optimizer is optimizer
    assert isinstance(conflearn_trainer.loss_fn, torch.nn.CrossEntropyLoss)
    assert isinstance(conflearn_trainer.conformal_loss_fn, UncertaintyAwareLoss)
    assert conflearn_trainer.mu == 0.2
    assert conflearn_trainer.alpha == 0.1
    assert conflearn_trainer.device == 'cpu'

    optimizer = optim.Adam(mock_model.parameters(), lr=0.1)
    conflearn_trainer = UncertaintyAwareTrainer(mock_model, optimizer, loss_fn=torch.nn.L1Loss(), mu=0.5, alpha=0.4,
                                                device='cuda')
    assert conflearn_trainer.model is mock_model
    assert conflearn_trainer.optimizer is optimizer
    assert isinstance(conflearn_trainer.loss_fn, torch.nn.L1Loss)
    assert isinstance(conflearn_trainer.conformal_loss_fn, UncertaintyAwareLoss)
    assert conflearn_trainer.mu == 0.5
    assert conflearn_trainer.alpha == 0.4
    assert conflearn_trainer.device == 'cuda'


def test_calculate_loss(mock_conflearn_trainer):
    output = torch.rand((100, 2))
    target = torch.randint(0, 2, (100,))
    Z_batch = torch.randint(0, 2, (100,))
    torch.manual_seed(42)
    loss = mock_conflearn_trainer.calculate_loss(output, target, Z_batch)
    torch.manual_seed(42)
    idx_ce = torch.where(Z_batch == 0)[0]
    loss_ce = mock_conflearn_trainer.loss_fn(output[idx_ce], target[idx_ce])
    loss_scores = mock_conflearn_trainer.conformal_loss_fn(output, target, Z_batch)
    except_loss = loss_ce + loss_scores * 0.2
    assert loss.item() == except_loss.item()

    torch.manual_seed(42)
    loss = mock_conflearn_trainer.calculate_loss(output, target, None, False)
    torch.manual_seed(42)
    Z_batch = torch.ones(len(output)).long()
    loss_ce = mock_conflearn_trainer.loss_fn(output, target)
    loss_scores = mock_conflearn_trainer.conformal_loss_fn(output, target, Z_batch)
    except_loss = loss_ce + loss_scores * 0.2
    assert loss.item() == except_loss.item()


def test_train_epoch(mock_conflearn_trainer, train_loader):
    train_loader = mock_conflearn_trainer.split_dataloader(train_loader)
    mock_conflearn_trainer.train_epoch(train_loader)


def test_validate(mock_conflearn_trainer, val_loader):
    torch.manual_seed(42)
    metrics = mock_conflearn_trainer.validate(val_loader)

    torch.manual_seed(42)
    except_loss_val = 0
    except_acc_val = 0
    for X_batch, Y_batch in val_loader:
        output = X_batch
        pred = output.argmax(dim=1)

        loss = mock_conflearn_trainer.calculate_loss(output, Y_batch, None, training=False)
        except_loss_val += loss

        acc = pred.eq(Y_batch).sum()
        except_acc_val += acc.item()

    except_loss_val /= len(val_loader)
    except_acc_val /= len(val_loader)

    assert metrics['val_loss'] == except_loss_val
    assert metrics['val_acc'] == except_acc_val


def test_train(mock_conflearn_trainer, train_loader, val_loader):
    save_dir = '.cache/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "test_conflearn_trainer.pt")
    mock_conflearn_trainer.train(train_loader, val_loader, 5, save_path)

    assert os.path.exists(save_path)
    os.remove(save_path)


def test_split_dataloader(mock_conflearn_trainer, train_loader):
    torch.manual_seed(42)
    train_loader = mock_conflearn_trainer.split_dataloader(train_loader)

    torch.manual_seed(42)
    dataset = train_loader.dataset
    X_data = dataset.X_data
    Y_data = dataset.Y_data

    Z_data = torch.zeros(len(dataset)).long()
    split = int(len(dataset) * 0.8)
    Z_data[torch.randperm(len(dataset))[split:]] = 1
    train_dataset = TrainDataset(X_data, Y_data, Z_data)
    except_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=True)

    assert torch.equal(train_loader.dataset.X_data, except_loader.dataset.X_data)
    assert torch.equal(train_loader.dataset.Y_data, except_loader.dataset.Y_data)
    assert torch.equal(train_loader.dataset.Z_data, except_loader.dataset.Z_data)
    assert train_loader.batch_size == except_loader.batch_size
    assert train_loader.drop_last == except_loader.drop_last
