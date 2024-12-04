# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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


def build_regression_model(model_name="NonLinearNet"):
    if model_name == "NonLinearNet":
        return NonLinearNet
    elif model_name == "NonLinearNet_with_Softmax":
        return Softmax
    else:
        raise NotImplementedError
