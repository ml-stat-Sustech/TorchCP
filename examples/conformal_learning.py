# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from torchcp.classification.trainer import ConfLearnTrainer
from .utils import Model_Ex1


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
          return x, torch.cat([z2,z3],1)
        else:
          return x
        
class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data, z_data):
        self.X_data = X_data
        self.y_data = y_data
        self.z_data = z_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.z_data[index]

    def __len__ (self):
        return len(self.X_data)

def setup_data_and_model(device):
    p = 100         # Number of features
    K = 6           # Number of possible labels
    delta_1 = 0
    delta_2 = 0.2
    a = 1

    n_train = 4800  # Number of data samples
    n_hout = 2000   # Number of hold out samples
    n_test = 2000   # Number of test samples

    data_model = Model_Ex1(K, p, delta_1, delta_2, a)   # Data generating model

    X_train = data_model.sample_X(n_train)              # Generate the data features
    Y_train = data_model.sample_Y(X_train)              # Generate the data labels conditional on the features

    n_tr_score = int(n_train * 0.2)                     # Number of data samples for training the new loss
    X_tr_score = data_model.sample_X(n_tr_score)        # Generate the data features
    Y_tr_score = data_model.sample_Y(X_tr_score)        # Generate the data labels conditional on the features

    X_hout = data_model.sample_X(n_hout)                # Generate independent hold-out data
    Y_hout = data_model.sample_Y(X_hout)

    X_test = data_model.sample_X(n_test, test=True)     # Generate independent test data
    Y_test = data_model.sample_Y(X_test)

    X_augmented = np.concatenate((X_train, X_tr_score), 0)
    Y_augmented = np.concatenate((Y_train, Y_tr_score), 0)

    Z_train = np.zeros(len(Y_train))
    Z_tr_score = np.ones(len(Y_tr_score))
    Z_augmented = np.concatenate((Z_train, Z_tr_score), 0)


    # Initialize loader for training data 
    X_train = torch.from_numpy(X_augmented).float().to(device)
    Y_train = torch.from_numpy(Y_augmented).long().to(device)
    if np.sum(np.unique(Z_augmented) > 0) > 0:
        eval_conf_train = True
    else:
        eval_conf_train = False
    Z_train = torch.from_numpy(Z_augmented).long().to(device)

    train_dataset = ClassifierDataset(X_train, Y_train, Z_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize loader for hold-out data (if available)
    X_hout = torch.from_numpy(X_hout).float().to(device)
    Y_hout = torch.from_numpy(Y_hout).long().to(device)
    Z_hout = torch.ones(Y_hout.shape).long().to(device)
    val_dataset = ClassifierDataset(X_hout, Y_hout, Z_hout)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ClassNNet(num_features=p, num_classes=K, use_dropout=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return train_loader, val_loader, model, optimizer


if __name__ == '__main__':
    alpha = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed=0)

    batch_size = 750
    lr = 0.001
    mu = 0.2

    train_loader, val_loader, model, optimizer = setup_data_and_model(device)

    conf_trainer = ConfLearnTrainer(model, optimizer, device)
    conf_trainer.train(train_loader, val_loader, num_epochs=4000)
    