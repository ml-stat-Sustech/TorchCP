# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from torchcp.classification.trainer import ConfLearnTrainer
from examples.utils import get_others_dir
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS


# Data generating model for conformal learning
class Model_Ex1:
    def __init__(self, K, p, delta_1, delta_2, a=1, device='cpu'):
        self.K = K
        self.p = p
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.a = a
        self.device = device

    def sample_X(self, n, test=False):
        X = torch.rand((n, self.p))
        if test:
            X[:, 6] = torch.empty(n).uniform_(0, self.a)
        return X.float().to(self.device)

    def compute_prob(self, X):
        X = X[:, 2:]
        prob = torch.zeros((X.shape[0], self.K))
        for i in range(X.shape[0]):
            if (X[i, 0] < self.delta_1):
                prob[i, :] = 1.0 / self.K
            else:
                K_half = torch.ceil(torch.tensor(self.K / 2)).int()
                if (X[i, 2] < 0.5):
                    if (X[i, 4] < self.delta_2):
                        prob[i, 0:K_half] = 1.0 / K_half
                    else:
                        idx = torch.round(self.K * X[i, 10] - 0.5).int()
                        prob[i, idx] = 1
                else:
                    if (X[i, 4] < self.delta_2):
                        prob[i, K_half:self.K] = 1.0 / (self.K - K_half)
                    else:
                        idx = torch.round(self.K * X[i, 10] - 0.5).int()
                        prob[i, idx] = 1

        prob_y = prob / prob.sum(dim=1, keepdim=True)
        return prob_y

    def sample_Y(self, X):
        prob_y = self.compute_prob(X)
        y = torch.multinomial(prob_y, num_samples=1, replacement=True)
        return y.flatten().long().to(self.device)


class ClassNNet(nn.Module):
    def __init__(self, num_features, num_classes, use_dropout=False):
        super(ClassNNet, self).__init__()

        self.use_dropout = use_dropout

        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, num_classes)

        self.z_dim = 256 + 128

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x, extract_features=False):
        x = self.layer_1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm1(x)

        z2 = self.layer_2(x)
        x = self.relu(z2)
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm2(x)

        z3 = self.layer_3(x)
        x = self.relu(z3)
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm3(x)
        x = self.layer_4(x)
        x = self.relu(x)

        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm4(x)

        x = self.layer_5(x)

        if extract_features:
            return x, torch.cat([z2, z3], 1)
        else:
            return x


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data, z_data):
        self.X_data = X_data
        self.y_data = y_data
        self.z_data = z_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.z_data[index]

    def __len__(self):
        return len(self.X_data)


class CommonDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return len(self.X_data)


class Oracle:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.sample_Y(X)

    def predict_proba(self, X):
        if (len(X.shape) == 1):
            X = X.reshape((1, X.shape[0]))
        prob = self.model.compute_prob(X)
        prob = torch.clamp(prob, min=1e-6, max=1.0)
        prob = prob / prob.sum(axis=1)[:, None]
        return prob


def difficulty_oracle(sets_oracle, size_cutoff=1):
    size_oracle = torch.sum(sets_oracle, dim=1)
    easy_idx = torch.where(size_oracle <= size_cutoff)[0]
    hard_idx = torch.where(size_oracle > size_cutoff)[0]
    return easy_idx, hard_idx


def evaluate_predictions(trainer, pred_sets, test_loader, labels, easy_idx=None, hard_idx=None, conditional=True):
    # Accuracy of Trainer
    y_pred = trainer.predict(test_loader)
    accuracy = torch.mean((y_pred != labels).float()).item() * 100

    # Marginal Coverage and Size
    marg_coverage = torch.mean(pred_sets[torch.arange(pred_sets.shape[0]), labels].float()).item()
    size = torch.mean(torch.sum(pred_sets, dim=1).float()).item()

    if conditional:
        y_hard = labels[hard_idx]
        S_easy = pred_sets[easy_idx]
        S_hard = pred_sets[hard_idx]

        # Evaluate conditional coverage
        wsc_coverage = torch.mean(S_hard[torch.arange(S_hard.shape[0]), y_hard].float()).item()

        # Evaluate conditional size
        size_easy = torch.mean(torch.sum(S_easy, dim=1).float()).item()
        size_hard = torch.mean(torch.sum(S_hard, dim=1).float()).item()
    else:
        wsc_coverage = None
        size_easy = None
        size_hard = None

    # Combine results
    out = pd.DataFrame({'Accuracy': [accuracy], 'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Size': [size], 'Size-hard': [size_hard], 'Size-easy': [size_easy]})
    return out


def setup_data_and_model(device):
    p = 100         # Number of features
    K = 6           # Number of possible labels
    delta_1 = 0
    delta_2 = 0.2
    a = 1

    n_train = 4800  # Number of data samples
    n_hout = 2000   # Number of hold out samples
    n_calib = 10000  # Number of calibration samples
    n_test = 2000   # Number of test samples

    data_model = Model_Ex1(K, p, delta_1, delta_2, a, device)   # Data generating model

    # Generate the data features
    X_train = data_model.sample_X(n_train)
    # Generate the data labels conditional on the features
    Y_train = data_model.sample_Y(X_train)

    # Number of data samples for training the new loss
    n_tr_score = int(n_train * 0.2)
    # Generate the data features
    X_tr_score = data_model.sample_X(n_tr_score)
    # Generate the data labels conditional on the features
    Y_tr_score = data_model.sample_Y(X_tr_score)

    # Generate independent hold-out data
    X_hout = data_model.sample_X(n_hout)
    Y_hout = data_model.sample_Y(X_hout)

    # Generate independent calibration data
    X_calib = data_model.sample_X(n_calib)
    Y_calib = data_model.sample_Y(X_calib)

    # Generate independent test data
    X_test = data_model.sample_X(n_test, test=True)
    Y_test = data_model.sample_Y(X_test)

    X_augmented = torch.cat((X_train, X_tr_score), 0)
    Y_augmented = torch.cat((Y_train, Y_tr_score), 0)

    Z_train = torch.zeros(len(Y_train))
    Z_tr_score = torch.ones(len(Y_tr_score))
    Z_augmented = torch.cat((Z_train, Z_tr_score), 0).long().to(device)

    # Initialize loader for training data
    train_dataset = ClassifierDataset(X_augmented, Y_augmented, Z_augmented)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize loader for hold-out data (if available)
    Z_hout = torch.ones(Y_hout.shape).long().to(device)
    val_dataset = ClassifierDataset(X_hout, Y_hout, Z_hout)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize loader for calibration data
    cal_dataset = CommonDataset(X_calib, Y_calib)
    cal_loader = DataLoader(cal_dataset, batch_size=100,
                            shuffle=True, drop_last=True)

    # Initialize loader for test data
    test_dataset = CommonDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=100,
                             shuffle=True, drop_last=True)

    oracle = Oracle(data_model)

    model = ClassNNet(num_features=p, num_classes=K,
                      use_dropout=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return train_loader, val_loader, cal_loader, test_loader, X_test, Y_test, oracle, model, optimizer


if __name__ == '__main__':
    alpha = 0.1
    batch_size = 750
    lr = 0.001
    mu = 0.2
    checkpoint_path = os.path.join(get_others_dir(), "conflearn")
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    set_seed(seed=42)

    #######################################
    # Loading dataset, a model and Conformal Learning Trainer
    #######################################
    train_loader, val_loader, cal_loader, test_loader, X_test, Y_test, oracle, model, optimizer = setup_data_and_model(
        device)
    conflearn_trainer = ConfLearnTrainer(model, optimizer, device=device)
    
    #######################################
    # Conformal Learning
    #######################################
    conflearn_trainer.train(train_loader, val_loader,
                            checkpoint_path=checkpoint_path, num_epochs=4000)

    # For early stopping loss
    conflearn_trainer_loss = ConfLearnTrainer(model, optimizer, device=device)
    conflearn_trainer_loss.load_checkpoint(checkpoint_path, "loss")

    # For early stopping acc
    conflearn_trainer_acc = ConfLearnTrainer(model, optimizer, device=device)
    conflearn_trainer_acc.load_checkpoint(checkpoint_path, "acc")

    #######################################
    # Evaluation for Conformal Learning
    #######################################

    # Oracle results and recognize hard samples
    sc_method_oracle = SplitPredictor(APS(score_type="identity"))
    sc_method_oracle._device = device
    pred_prob_oracle = oracle.predict_proba(X_test)
    sets_oracle = sc_method_oracle.predict_with_logits(
        pred_prob_oracle, 1 - alpha)
    easy_idx, hard_idx = difficulty_oracle(sets_oracle)

    # Results of Split Conformal Prediction for Conformal Learning
    black_boxes = [conflearn_trainer,
                   conflearn_trainer_loss, conflearn_trainer_acc]

    results = pd.DataFrame()
    for i in range(len(black_boxes)):
        sc_method = SplitPredictor(APS(), black_boxes[i].model)
        sc_method.calibrate(cal_loader, alpha)
        pred_sets = sc_method.predict(X_test)

        res = evaluate_predictions(black_boxes[i], pred_sets, test_loader, Y_test, easy_idx, hard_idx, conditional=True)
        results = pd.concat([results, res])

    print(results)
