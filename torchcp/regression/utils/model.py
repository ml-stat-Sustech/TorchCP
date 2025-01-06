# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn


class NonLinearNet(nn.Module):
    """
    Non-linear neural network with ReLU activations and dropout.

    This class defines a simple non-linear neural network architecture with 
    multiple hidden layers using ReLU activations and dropout for regularization.

    Args:
        input_dim (int): Dimensionality of the input layer.
        output_dim (int): Dimensionality of the output layer.
        hidden_size (int): Number of neurons in each hidden layer.
        dropout (float): Dropout rate for regularization.

    """

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
        """
        Forward pass of the non-linear network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the network.

        """
        return self.base_model(x)


class Softmax(nn.Module):
    """
    Neural network with Softmax output layer.

    This class builds upon the `NonLinearNet` class and adds a Softmax 
    activation layer at the output for probability distributions.

    Args:
        input_dim (int): Dimensionality of the input layer.
        output_dim (int): Dimensionality of the output layer.
        hidden_size (int): Number of neurons in each hidden layer.
        dropout (float): Dropout rate for regularization.

    """

    def __init__(self, input_dim, output_dim, hidden_size, dropout):
        super(Softmax, self).__init__()
        self.base_model = nn.Sequential(
            NonLinearNet(input_dim, output_dim, hidden_size, dropout),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Forward pass of the network with Softmax output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output probability distribution.

        """
        return self.base_model(x)


class EncodingNetwork(nn.Module):
    """
    Encoding network with time embedding.

    This class defines a network for encoding input data while incorporating 
    positional information through a time embedding.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden layer.
        output_dim (int): Dimensionality of the encoded output.
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.

    """

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
        """
        Forward pass of the encoding network.

        Args:
            Z_t (torch.Tensor): Input data at time step t.
            t (int): Current time step.
            T (int): Total number of time steps.

        Returns:
            torch.Tensor: Encoded representation of the input data with time information.

        """
        mL_Z_t = self.mL(Z_t)
        z_time = (t / T).unsqueeze(1) 
        encoded = torch.cat([mL_Z_t, z_time], dim=1) 
        return encoded


class HopfieldAssociation(nn.Module):
    """
    Hopfield-like association network.

    This class implements a simplified Hopfield-like network for 
    calculating association scores between input vectors. 

    Args:
        input_dim (int): Dimensionality of the input vectors.
        hidden_dim (int): Dimensionality of the hidden layer.
        temperature (float, optional): Temperature parameter for the Softmax function. 
                                        Defaults to 1.

    """

    def __init__(self, input_dim, hidden_dim, temperature=1):
        super(HopfieldAssociation, self).__init__()
        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, Z_t):
        """
        Calculates association scores between input vectors.

        Args:
            Z_t (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Association matrix of shape (batch_size, batch_size), 
                          where element (i, j) represents the association score 
                          between the i-th and j-th input vectors.

        This function first projects the input vectors into a hidden space using 
        two separate linear transformations (`W_q` and `W_k`). 
        Then, it computes the dot product between the transformed vectors 
        and applies a Softmax function to obtain association scores. 
        The temperature parameter (`beta`) controls the sharpness of the 
        Softmax distribution.

        """
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
