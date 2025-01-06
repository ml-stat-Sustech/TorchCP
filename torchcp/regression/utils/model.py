# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class NonLinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout):
        super(NonLinearNet, self).__init__()
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.base_model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.output_dim),
        )

    def forward(self, x):
        return self.base_model(x)


class Softmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout):
        super(Softmax, self).__init__()
        self.base_model = nn.Sequential(
            NonLinearNet(input_dim, output_dim, hidden_size, dropout),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.base_model(x)

class EncodingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0):
        super(EncodingNetwork, self).__init__()
        self.mL = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, Z_t, t, T):
        mL_Z_t = self.mL(Z_t) 
        z_time = (t / T).unsqueeze(1) 
        encoded = torch.cat([mL_Z_t, z_time], dim=1) 
        return encoded


class HopfieldAssociation(nn.Module):
    def __init__(self, input_dim, hidden_dim, temperature=1):
        super(HopfieldAssociation, self).__init__()
        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))


    def forward(self, Z_t):
        Q = self.W_q(Z_t)  # Query
        K = self.W_k(Z_t)  # Key
        A = torch.softmax(self.beta * torch.matmul(Q, K.T), dim=-1)
        return A


def build_regression_model(model_name="NonLinearNet"):
    if model_name == "NonLinearNet":
        return NonLinearNet
    elif model_name == "NonLinearNet_with_Softmax":
        return Softmax
    else:
        raise NotImplementedError
