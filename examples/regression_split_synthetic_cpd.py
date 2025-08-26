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
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from examples.regression_cqr_synthetic import prepare_dataset
from torchcp.regression.predictor import ConformalPredictiveDistribution 
from torchcp.regression.score import Sign
from torchcp.regression.utils import build_regression_model


if __name__ == "__main__":
    # get dataloader
    train_loader, cal_loader, test_loader = prepare_dataset(train_ratio=0.4, cal_ratio=0.2, batch_size=128)
    # build regression model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_regression_model("NonLinearNet")(next(iter(train_loader))[0].shape[1], 1, 64, 0.5).to(device)

    
    
    
    # train model
    epochs = 100
    criterion = nn.MSELoss()
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)    
        
    for tmp_x, tmp_y in train_loader:
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.reshape(-1, 1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # CPD
    predictor = ConformalPredictiveDistribution(score_function=Sign(), model=model)
    predictor.calibrate(cal_loader)

    x = next(iter(test_loader))[0].to(device)
    prediction_intervals = predictor.predict(x)
        
    print(prediction_intervals.shape)