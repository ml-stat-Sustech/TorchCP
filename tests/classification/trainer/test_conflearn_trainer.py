# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import shutil
import pytest
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchcp.classification.loss import ConfLearnLoss
from torchcp.classification.trainer import ConfLearnTrainer

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data, z_data):
        self.X_data = X_data
        self.y_data = y_data
        self.z_data = z_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.z_data[index]

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
    return ConfLearnTrainer(mock_model, optimizer)


@pytest.fixture
def train_loader():
    X_train = torch.rand((100, 3)).float()
    Y_train = torch.randint(0, 3, (100, )).long()
    Z_train = torch.randint(0, 2, (100, )).long()
    train_dataset = ClassifierDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(
        train_dataset, batch_size=20, shuffle=True, drop_last=True)
    return train_loader


@pytest.fixture
def val_loader():
    X_val = torch.rand((100, 3)).float()
    Y_val = torch.randint(0, 3, (100, )).long()
    Z_val = torch.randint(0, 2, (100, )).long()
    val_dataset = ClassifierDataset(X_val, Y_val, Z_val)
    val_loader = DataLoader(
        val_dataset, batch_size=20, shuffle=True, drop_last=True)
    return val_loader


def test_initialization(mock_model):
    optimizer = optim.Adam(mock_model.parameters(), lr=0.1)
    conflearn_trainer = ConfLearnTrainer(mock_model, optimizer)
    assert conflearn_trainer.model is mock_model
    assert conflearn_trainer.optimizer is optimizer
    assert isinstance(conflearn_trainer.criterion_pred_loss_fn, torch.nn.CrossEntropyLoss)
    assert isinstance(conflearn_trainer.conformal_loss_fn, ConfLearnLoss)
    assert conflearn_trainer.mu == 0.2
    assert conflearn_trainer.alpha == 0.1
    assert conflearn_trainer.device == 'cpu'

    optimizer = optim.Adam(mock_model.parameters(), lr=0.1)
    conflearn_trainer = ConfLearnTrainer(mock_model, optimizer, criterion_pred_loss_fn=torch.nn.L1Loss(), mu=0.5, alpha=0.4, device='cuda')
    assert conflearn_trainer.model is mock_model
    assert conflearn_trainer.optimizer is optimizer
    assert isinstance(conflearn_trainer.criterion_pred_loss_fn, torch.nn.L1Loss)
    assert isinstance(conflearn_trainer.conformal_loss_fn, ConfLearnLoss)
    assert conflearn_trainer.mu == 0.5
    assert conflearn_trainer.alpha == 0.4
    assert conflearn_trainer.device == 'cuda'


def test_calculate_loss(mock_conflearn_trainer):

    output = torch.rand((100, 2))
    target = torch.randint(0, 2, (100, ))
    Z_batch = torch.randint(0, 2, (100, ))
    torch.manual_seed(42)
    loss = mock_conflearn_trainer.calculate_loss(output, target, Z_batch)
    torch.manual_seed(42)
    idx_ce = torch.where(Z_batch == 0)[0]
    loss_ce = mock_conflearn_trainer.criterion_pred_loss_fn(output[idx_ce], target[idx_ce])
    loss_scores = mock_conflearn_trainer.conformal_loss_fn(output, target, Z_batch)
    except_loss = loss_ce + loss_scores * 0.2
    assert loss.item() == except_loss.item()

    torch.manual_seed(42)
    loss = mock_conflearn_trainer.calculate_loss(output, target, Z_batch, False)
    torch.manual_seed(42)
    loss_ce = mock_conflearn_trainer.criterion_pred_loss_fn(output, target)
    loss_scores = mock_conflearn_trainer.conformal_loss_fn(output, target, Z_batch)
    except_loss = loss_ce + loss_scores * 0.2
    assert loss.item() == except_loss.item()


def test_train_epoch(mock_conflearn_trainer, train_loader):

    mock_conflearn_trainer.train_epoch(train_loader)


def test_validate(mock_conflearn_trainer, val_loader):

    _, acc_val = mock_conflearn_trainer.validate(val_loader)

    except_acc_val = 0

    for X_batch, Y_batch, _ in val_loader:
        output = X_batch
        pred = output.argmax(dim=1)
        acc = pred.eq(Y_batch).sum()
        except_acc_val += acc.item()

    except_acc_val /= len(val_loader)
    assert acc_val == except_acc_val


def test_train(mock_conflearn_trainer, train_loader, val_loader):
    save_dir = '.cache/test_conflearn_trainer/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "trainer")
    mock_conflearn_trainer.train(train_loader, save_path, val_loader, num_epochs=5)

    assert os.path.exists(save_path + "loss.pt")
    assert os.path.exists(save_path + "acc.pt")
    assert os.path.exists(save_path + "final.pt")

    shutil.rmtree(save_dir)


def test_save_and_load_checkpoint(mock_conflearn_trainer):
    save_dir = '.cache/test_conflearn_trainer/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "trainer")
    mock_conflearn_trainer.save_checkpoint(0, save_path, 'final')
    assert os.path.exists(save_path + "final.pt")

    checkpoint = torch.load(save_path + "final.pt")
    assert checkpoint['epoch'] == 0
    assert checkpoint['model_state_dict'] == mock_conflearn_trainer.model.state_dict()
    assert checkpoint['optimizer_state_dict'] == mock_conflearn_trainer.optimizer.state_dict()

    model = MockModel()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    conflearn_trainer = ConfLearnTrainer(model, optimizer)
    conflearn_trainer.load_checkpoint(save_path, 'final')
    assert conflearn_trainer.model.state_dict() == mock_conflearn_trainer.model.state_dict()
    assert conflearn_trainer.optimizer.state_dict() == mock_conflearn_trainer.optimizer.state_dict()

    conflearn_trainer.load_checkpoint(save_path, 'loss')
    assert conflearn_trainer.model.state_dict() == mock_conflearn_trainer.model.state_dict()
    assert conflearn_trainer.optimizer.state_dict() == mock_conflearn_trainer.optimizer.state_dict()

    shutil.rmtree(save_dir)