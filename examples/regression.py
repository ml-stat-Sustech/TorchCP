import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

from torchcp.regression.predictors import SplitPredictor, CQR, R2CCP
from torchcp.regression.loss import QuantileLoss, R2ccpLoss
from torchcp.utils import fix_randomness
from examples.common.dataset import build_reg_data
from examples.common.utils import build_regression_model


def train(model, device, epoch, train_data_loader, criterion, optimizer):
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader):
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.unsqueeze(dim=1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def calculate_midpoints(train_loader, cal_loader, K):
    all_data = []
    for loader in [train_loader, cal_loader]:
        for data in loader:
            all_data.append(data[0])
    all_data_tensor = torch.cat(all_data, dim=0)

    min_value = torch.min(all_data_tensor)
    max_value = torch.max(all_data_tensor)
    midpoints = torch.linspace(min_value, max_value, steps=K)

    return midpoints

if __name__ == '__main__':
    ##################################
    # Preparing dataset
    ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fix_randomness(seed=1)
    X, y = build_reg_data()
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index1 = int(len(indices) * 0.4)
    split_index2 = int(len(indices) * 0.6)
    part1, part2, part3 = np.split(indices, [split_index1, split_index2])
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[part1, :])
    train_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part1, :])), torch.from_numpy(y[part1]))
    cal_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part2, :])), torch.from_numpy(y[part2]))
    test_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part3, :])), torch.from_numpy(y[part3]))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

    epochs = 100
    alpha = 0.1

    ##################################
    # Split Conformal Prediction
    ##################################
    print("########################## SplitPredictor ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 1, 64, 0.5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)

    model.eval()
    predictor = SplitPredictor(model)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader)) 

    ##################################
    # Conformal Quantile Regression
    ##################################
    print("########################## CQR ###########################")

    quantiles = [alpha / 2, 1 - alpha / 2]
    model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)

    model.eval()
    predictor = CQR(model)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))

    ##################################
    # Conformal Prediction via Regression-as-Classification
    ##################################
    print("########################## R2CCP ###########################")

    train_data_loader_r2ccp = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    cal_data_loader_r2ccp = torch.utils.data.DataLoader(cal_dataset, batch_size=32, shuffle=False, pin_memory=True)
    test_data_loader_r2ccp = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    p = 0.5
    tau = 0.2
    K = 50

    midpoints = calculate_midpoints(train_data_loader_r2ccp, cal_data_loader_r2ccp, K)
    model = build_regression_model("Softmax")(X.shape[1], K, 1000, 0).to(device)
    criterion = R2ccpLoss(p, tau, K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader_r2ccp, criterion, optimizer)

    model.eval()
    predictor = R2CCP(model, K)
    predictor.calibrate(cal_data_loader_r2ccp, alpha, midpoints)
    print(predictor.evaluate(test_data_loader_r2ccp, midpoints))