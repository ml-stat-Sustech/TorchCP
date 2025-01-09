# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from transformers import set_seed

from examples.utils import build_reg_data
from torchcp.regression.loss import QuantileLoss
from torchcp.regression.predictor import ACIPredictor
from torchcp.regression.score import CQR
from torchcp.regression.utils import build_regression_model


def prepare_aci_dataset(train_ratio=0.5, batch_size=100):
    """
    Prepare datasets for Adaptive Conformal Inference.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        train_ratio (float): Ratio of training data
        batch_size (int): Batch size for data loaders
    
    Returns:
        tuple: Training and test data loaders
    """
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
    lookback = 200  # number of historical data points
    score_function = CQR()
    predictor = ACIPredictor(score_function=score_function, model=model, gamma=0.005)
    
    # train regression model
    ## The Train function is required here, and the user can customize the training parameters
    criterion = QuantileLoss([alpha / 2, 1 - alpha / 2])
    predictor.train(train_loader, alpha=alpha, criterion=criterion, epochs=100, lr=0.01, verbose=True)
    
    # generate conformal prediction interval
    predict_list = []
    train_dataset = train_loader.dataset
    samples = [train_dataset[i] for i in range(len(train_dataset) - lookback, len(train_dataset))]
    x_lookback = torch.stack([sample[0] for sample in samples]).to(device)
    y_lookback = torch.stack([sample[1] for sample in samples]).to(device)
    pred_interval_lookback = predictor.predict(x_lookback)
    
    for tmp_x, tmp_y in test_loader:
        tmp_x, tmp_y = tmp_x.to(device), tmp_y.to(device)
        tmp_prediction_intervals = predictor.predict(x_batch=tmp_x, x_lookback=x_lookback, y_lookback=y_lookback,
                                                    pred_interval_lookback=pred_interval_lookback,
                                                    train=True, update_alpha=True)
        predict_list.append(tmp_prediction_intervals)
        pred_interval_lookback = torch.cat([pred_interval_lookback, tmp_prediction_intervals], dim=0)[-lookback:]
        x_lookback = torch.cat([x_lookback, tmp_x], dim=0)[-lookback:]
        y_lookback = torch.cat([y_lookback, tmp_y], dim=0)[-lookback:]
            
    predicts_interval = torch.cat(predict_list, dim=0).to(device)
    
    # evaluate on test dataloader
    result = predictor.evaluate(test_loader)
    print(result)