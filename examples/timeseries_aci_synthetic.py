# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from examples.utils import build_reg_data
from torchcp.regression.predictor import ACIPredictor
from torchcp.regression.score import CQR
from torchcp.regression.utils import build_regression_model


def prepare_aci_dataset(train_ratio=0.5, batch_size=100):
    # construct datasets
    X, y = build_reg_data(data_name="synthetic")
    # Split indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(len(indices) * train_ratio)
    train_indices, test_indices = np.split(indices, [split_index])

    # Scale features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[train_indices, :])

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[train_indices, :])),
        torch.from_numpy(y[train_indices])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[test_indices, :])),
        torch.from_numpy(y[test_indices])
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == "__main__":
    # get dataloader
    train_loader, test_loader = prepare_aci_dataset(train_ratio=0.5, batch_size=128)
    # build regression model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_regression_model("NonLinearNet")(next(iter(train_loader))[0].shape[1], 2, 64, 0.5).to(device)
    
    # CP
    alpha = 0.1  # confidence level
    predictor = ACIPredictor(score_function=CQR(), model=model, gamma=0.005)
    
    # Step1: train regression model
    ## The Train function is required here, and the user can customize the training parameters
    predictor.train(train_loader, alpha=alpha, epochs=100, lr=0.01, verbose=True)
    
    # Step2: prediction
    #### Option1: generate conformal prediction interval for x_batch
    x = next(iter(test_loader))[0].to(device)
    prediction_intervals = predictor.predict(x)
    
    #### Option2: generate conformal prediction interval for x_batch using history data
    lookback = 200  # number of historical data points
    samples = [train_loader.dataset[i] for i in range(len(train_loader.dataset) - lookback, len(train_loader.dataset))]
    x_lookback = torch.stack([sample[0] for sample in samples]).to(device)
    y_lookback = torch.stack([sample[1] for sample in samples]).to(device)
    
    x = next(iter(test_loader))[0].to(device)
    prediction_intervals = predictor.predict(x_batch=x, x_lookback=x_lookback, y_lookback=y_lookback)
    
    # Step3: evaluate conformal prediction on test dataloader
    result = predictor.evaluate(test_loader)
    print(result)