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
from torchcp.regression.loss import QuantileLoss
from torchcp.regression.predictor import SplitPredictor
from torchcp.regression.score import CQR
from torchcp.regression.utils import build_regression_model


def prepare_dataset(train_ratio=0.4, cal_ratio=0.2, batch_size=128):
    # construct datasets
    X, y = build_reg_data(data_name="synthetic")
    
    # Split indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index1 = int(len(indices) * train_ratio)
    split_index2 = int(len(indices) * (train_ratio + cal_ratio))
    part1, part2, part3 = np.split(indices, [split_index1, split_index2])

    # Scale features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[part1, :])

    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[part1, :])),
        torch.from_numpy(y[part1])
    )
    cal_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[part2, :])),
        torch.from_numpy(y[part2])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(scalerX.transform(X[part3, :])),
        torch.from_numpy(y[part3])
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    cal_loader = torch.utils.data.DataLoader(
        cal_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, cal_loader, test_loader


if __name__ == "__main__":
    # get dataloader
    train_loader, cal_loader, test_loader = prepare_dataset(train_ratio=0.4, cal_ratio=0.2, batch_size=128)
    # build regression model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_regression_model("NonLinearNet")(next(iter(train_loader))[0].shape[1], 2, 64, 0.5).to(device)
    
    # CP
    alpha = 0.1  # confidence level
    score_function = CQR()
    predictor = SplitPredictor(score_function=score_function, model=model)
    
    # train regression model
    ## We've provided an example function here to help with model training, 
    ## or user can simply enter the trained model in predictor and ignore this code
    criterion = QuantileLoss([alpha / 2, 1 - alpha / 2])
    predictor.train(train_loader, criterion=criterion, alpha=alpha, epochs=100, lr=0.01, verbose=True)
    
    # calibrate
    predictor.calibrate(cal_dataloader=cal_loader, alpha=alpha)
    
    # generate conformal prediction interval
    predict_list = []
    with torch.no_grad():
        for tmp_x, _ in test_loader:
            tmp_x = tmp_x.to(device)
            tmp_prediction_intervals = predictor.predict(tmp_x)
            predict_list.append(tmp_prediction_intervals)
    predicts_interval = torch.cat(predict_list, dim=0).to(device)
    
    # evaluate on test dataloader
    result = predictor.evaluate(test_loader)
    print(result)
    
    
    
    
    