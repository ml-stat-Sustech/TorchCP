import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler


from torchcp.regression.predictors import SplitPredictor, CQR, ACI, R2CCP
from torchcp.regression.loss import QuantileLoss, R2ccpLoss
from torchcp.regression import Metrics 
from torchcp.utils import fix_randomness
from utils import build_reg_data, build_regression_model
from torchcp.regression.utils import calculate_midpoints


def train(model, device, epoch, train_data_loader, criterion, optimizer):
    for index, (tmp_x, tmp_y) in enumerate(train_data_loader):
        outputs = model(tmp_x.to(device))
        loss = criterion(outputs, tmp_y.unsqueeze(dim=1).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_SplitPredictor():
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
    # Conformal Prediction via Regression-as-Classification
    ##################################
    print("########################## R2CCP ###########################")

    p = 0.5
    tau = 0.2
    K = 50

    train_and_cal_dataset = ConcatDataset([train_dataset, cal_dataset])
    train_and_cal_data_loader = torch.utils.data.DataLoader(train_and_cal_dataset, batch_size=100, shuffle=True,
                                                            pin_memory=True)
    midpoints = calculate_midpoints(train_and_cal_data_loader, K)

    model = build_regression_model("NonLinearNet_with_Softmax")(X.shape[1], K, 1000, 0).to(device)
    criterion = R2ccpLoss(p, tau, midpoints)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)

    model.eval()
    predictor = R2CCP(model, midpoints)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))


def test_time_series():
    ##################################
    # Preparing dataset
    ##################################
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    fix_randomness(seed=2)
    X, y = build_reg_data(data_name="synthetic")
    num_examples = X.shape[0]
    T0 = int(num_examples * 0.4)
    train_dataset = TensorDataset(torch.from_numpy(X[:T0, :]), torch.from_numpy(y[:T0]))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)

    alpha = 0.1
    quantiles = [alpha / 2, 1 - alpha / 2]
    model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in range(epochs):
        train(model, device, epoch, train_data_loader, criterion, optimizer)
            
    model.eval()        
    ##################################
    # Conformal Quantile Regression
    ##################################
    print("########################## CQR ###########################")

    predictor = CQR(model)
    cal_dataset = TensorDataset(torch.from_numpy(X[0:T0, :]), torch.from_numpy(y[0:T0]))
    test_dataset = TensorDataset(torch.from_numpy(X[T0:, :]), torch.from_numpy(y[T0:]))
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
    test_data_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))      

    ##################################
    # Adaptive Conformal Inference,
    ##################################      
    print("########################## ACI ###########################")
    predictor = ACI(model, 0.005)
    test_y = torch.from_numpy(y[T0:num_examples]).to(device)
    predicts = torch.zeros((num_examples - T0, 2)).to(device)
    for i in range(num_examples - T0):
        with torch.no_grad():
            cal_dataset = TensorDataset(torch.from_numpy(X[i:(T0 + i), :]), torch.from_numpy(y[i:(T0 + i)]))
            cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=100, shuffle=False, pin_memory=True)
            predictor.calibrate(cal_data_loader, alpha)
            tmp_x = torch.from_numpy(X[(T0 + i), :])
            if i == 0:
                tmp_prediction_intervals = predictor.predict(tmp_x)
            else:
                tmp_prediction_intervals = predictor.predict(tmp_x, test_y[i - 1], predicts[i - 1])
            predicts[i, :] = tmp_prediction_intervals

    metrics = Metrics()
    print("Evaluating prediction sets...")
    print(f"Coverage_rate: {metrics('coverage_rate')(predicts, test_y)}")
    print(f"Average_size: {metrics('average_size')(predicts)}")


