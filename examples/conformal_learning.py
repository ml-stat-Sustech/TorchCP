# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from torchcp.classification.trainer import ConfLearnTrainer
from examples.utils import Model_Ex1, get_others_dir
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS


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
        prob = np.clip(prob, 1e-6, 1.0)
        prob = prob / prob.sum(axis=1)[:, None]
        return prob


def difficulty_oracle(S_oracle, size_cutoff=1):
    size_oracle = np.array([len(S) for S in S_oracle])
    easy_idx = np.where(size_oracle <= size_cutoff)[0]
    hard_idx = np.where(size_oracle > size_cutoff)[0]
    return easy_idx, hard_idx


def evaluate_predictions(S, X, y, hard_idx=None, conditional=True, linear=False):
    # Marginal coverage
    marg_coverage = np.mean([y[i] in S[i] for i in range(len(y))])

    if conditional:
        y_hard = y[hard_idx]
        S_hard = [S[i] for i in hard_idx]

        # Evaluate conditional coverage
        wsc_coverage = np.mean([y_hard[i] in S_hard[i]
                               for i in range(len(y_hard))])

        # Evaluate conditional size
        size_hard = np.mean([len(S[i]) for i in hard_idx])
        size_easy = np.mean([len(S[i])
                            for i in range(len(y)) if i not in hard_idx])
        size_hard_median = np.median([len(S[i]) for i in hard_idx])
        size_easy_median = np.median(
            [len(S[i]) for i in range(len(y)) if i not in hard_idx])

        n_hard = len(hard_idx)
        n_easy = len(y) - len(hard_idx)

    else:
        wsc_coverage = None

    # Size and size conditional on coverage
    size = np.mean([len(S[i]) for i in range(len(y))])
    size_median = np.median([len(S[i]) for i in range(len(y))])
    idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
    size_cover = np.mean([len(S[i]) for i in idx_cover])
    # Combine results
    out = pd.DataFrame({'Coverage': [marg_coverage], 'Conditional coverage': [wsc_coverage],
                        'Size': [size], 'Size (median)': [size_median],
                        'Size-hard': [size_hard], 'Size-easy': [size_easy],
                        'Size-hard (median)': [size_hard_median], 'Size-easy (median)': [size_easy_median],
                        'n-hard': [n_hard], 'n-easy': [n_easy],
                        'Size conditional on cover': [size_cover]})
    return out


def eval_predictions(X, Y, box, data="unknown", printing=True):
    Y_pred = box.predict(X)
    class_error = np.mean(Y != Y_pred)
    if printing:
        print("Classification error on {:s} data: {:.1f}%".format(
            data, class_error * 100))
    return (class_error * 100)


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

    data_model = Model_Ex1(K, p, delta_1, delta_2, a)   # Data generating model

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

    X_augmented = np.concatenate((X_train, X_tr_score), 0)
    Y_augmented = np.concatenate((Y_train, Y_tr_score), 0)

    Z_train = np.zeros(len(Y_train))
    Z_tr_score = np.ones(len(Y_tr_score))
    Z_augmented = np.concatenate((Z_train, Z_tr_score), 0)

    # Initialize loader for training data
    X_train = torch.from_numpy(X_augmented).float().to(device)
    Y_train = torch.from_numpy(Y_augmented).long().to(device)
    Z_train = torch.from_numpy(Z_augmented).long().to(device)
    train_dataset = ClassifierDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize loader for hold-out data (if available)
    X_hout = torch.from_numpy(X_hout).float().to(device)
    Y_hout = torch.from_numpy(Y_hout).long().to(device)
    Z_hout = torch.ones(Y_hout.shape).long().to(device)
    val_dataset = ClassifierDataset(X_hout, Y_hout, Z_hout)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize loader for calibration data
    X_calib = torch.from_numpy(X_calib).float().to(device)
    Y_calib = torch.from_numpy(Y_calib).long().to(device)
    cal_dataset = CommonDataset(X_calib, Y_calib)
    cal_loader = DataLoader(cal_dataset, batch_size=100,
                            shuffle=True, drop_last=True)

    # Initialize loader for test data
    X_test = torch.from_numpy(X_test).float().to(device)
    Y_test = torch.from_numpy(Y_test).long().to(device)
    test_dataset = CommonDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=100,
                             shuffle=True, drop_last=True)

    oracle = Oracle(data_model)

    model = ClassNNet(num_features=p, num_classes=K,
                      use_dropout=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return train_loader, val_loader, cal_loader, X_test, Y_test, oracle, model, optimizer


if __name__ == '__main__':
    alpha = 0.1
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    set_seed(seed=42)

    batch_size = 750
    lr = 0.001
    mu = 0.2

    checkpoint_path = os.path.join(get_others_dir(), "conflearn")

    train_loader, val_loader, cal_loader, X_test, Y_test, oracle, model, optimizer = setup_data_and_model(
        device)

    conflearn_trainer = ConfLearnTrainer(model, optimizer, device=device)
    conflearn_trainer.train(train_loader, val_loader,
                            checkpoint_path=checkpoint_path, num_epochs=4000)

    # For early stopping loss
    conflearn_trainer_loss = ConfLearnTrainer(model, optimizer, device=device)
    if os.path.exists(checkpoint_path + "_loss"):
        conflearn_trainer_loss.load_checkpoint(checkpoint_path + "_loss")
    else:
        conflearn_trainer_loss.load_checkpoint(checkpoint_path + "_final")

    # For early stopping acc
    conflearn_trainer_acc = ConfLearnTrainer(model, optimizer, device=device)
    if os.path.exists(checkpoint_path + "_acc"):
        conflearn_trainer_loss.load_checkpoint(checkpoint_path + "_acc")
    else:
        conflearn_trainer_loss.load_checkpoint(checkpoint_path + "_final")

    black_boxes = [conflearn_trainer,
                   conflearn_trainer_loss, conflearn_trainer_acc]


    sc_methods = []
    for i in range(len(black_boxes)):
        sc_method = SplitPredictor(APS(), black_boxes[i].model)
        sc_method.calibrate(cal_loader, alpha)
        sc_methods.append(sc_method)

    results = pd.DataFrame()

    sc_method_oracle = SplitPredictor(APS(score_type="identity"))
    sc_method_oracle._device = device
    pred_prob_oracle = oracle.predict_proba(X_test.cpu().numpy())
    sets_oracle = sc_method_oracle.predict_with_logits(
        torch.tensor(pred_prob_oracle), 1 - alpha)
    row_indices, col_indices = sets_oracle.nonzero(as_tuple=True)
    sets_oracle = [col_indices[row_indices == i].tolist()
                   for i in range(sets_oracle.size(0))]

    easy_idx, hard_idx = difficulty_oracle(sets_oracle)


    for k in range(len(black_boxes)):
        sets = sc_methods[k].predict(X_test)
        row_indices, col_indices = sets.nonzero(as_tuple=True)
        sets = [col_indices[row_indices == i].tolist()
                for i in range(sets.size(0))]

        res = evaluate_predictions(
            sets, X_test, Y_test, hard_idx, conditional=True)
        # res['Error'] = eval_predictions(X_test, Y_test, black_boxes[k], data="test")

        results = pd.concat([results, res])

    results = results.reset_index()

    print(results)
