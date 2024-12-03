import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, ConcatDataset

from torchcp.regression.loss import QuantileLoss, R2ccpLoss
from torchcp.regression.predictors import SplitPredictor, EnsemblePredictor, ACIPredictor
from torchcp.regression.scores import split, CQR, CQRR, CQRM, CQRFM, R2CCP
from torchcp.regression.utils import calculate_midpoints, build_regression_model
from torchcp.utils import fix_randomness
from .utils import build_reg_data


def test_split_predictor():
    print("##########################################")
    print("######## Testing regression algorithm")
    print("##########################################")
    ##################################
    # Preparing dataset
    ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fix_randomness(seed=1)
    X, y = build_reg_data(data_name="synthetic")
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

    # CP parameters
    alpha = 0.1
    
    epochs = 20
    ##################################
    # Split Conformal Prediction
    ##################################
    print("########################## SplitPredictor ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 1, 64, 0.5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    score_function = split()
    predictor = SplitPredictor(score_function=score_function, model=model)
    predictor.fit(train_dataloader=train_data_loader, epochs=epochs, criterion=criterion, optimizer=optimizer)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))
    
    ##################################
    # Conformal Quantile Regression
    ##################################
    print("########################## Conformal Quantile Regression ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
    quantiles = [alpha / 2, 1 - alpha / 2]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    score_function = CQR()
    predictor = SplitPredictor(score_function, model=model)
    predictor.fit(train_dataloader=train_data_loader, epochs=epochs, criterion=criterion, optimizer=optimizer)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))
    
    ##################################
    # Conformal Quantile Regression-R
    ##################################
    print("########################## CQRR ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
    quantiles = [alpha / 2, 1 - alpha / 2]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    score_function = CQRR()
    predictor = SplitPredictor(score_function, model=model)
    predictor.fit(train_dataloader=train_data_loader, epochs=epochs, criterion=criterion, optimizer=optimizer)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))
    
    ##################################
    # Conformal Quantile Regression Median
    ##################################
    print("########################## CQRM ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 3, 64, 0.5).to(device)
    quantiles = [alpha / 2, 1 / 2, 1 - alpha / 2]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    score_function = CQRM()
    predictor = SplitPredictor(score_function, model=model)
    predictor.fit(train_dataloader=train_data_loader, epochs=epochs, criterion=criterion, optimizer=optimizer)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))
    
    ##################################
    # Conformal Quantile Regression Fraction Median
    ##################################
    print("########################## CQRFM ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 3, 64, 0.5).to(device)
    quantiles = [alpha / 2, 1 / 2, 1 - alpha / 2]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    score_function = CQRFM()
    predictor = SplitPredictor(score_function, model=model)
    predictor.fit(train_dataloader=train_data_loader, alpha=alpha)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))
    
    ##################################
    # Conformal Prediction via Regression-as-Classification
    ##################################
    print("########################## R2CCP ###########################")
    p = 0.5
    tau = 0.2
    K = 50
    model = build_regression_model("NonLinearNet_with_Softmax")(X.shape[1], K, 1000, 0).to(device)
    train_and_cal_dataset = ConcatDataset([train_dataset, cal_dataset])
    train_and_cal_data_loader = torch.utils.data.DataLoader(train_and_cal_dataset, batch_size=100, shuffle=True,
                                                            pin_memory=True)
    midpoints = calculate_midpoints(train_and_cal_data_loader, K).to(device)
    criterion = R2ccpLoss(p, tau, midpoints)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    score_function = R2CCP(midpoints)
    predictor = SplitPredictor(score_function, model=model)
    predictor.fit(train_dataloader=train_data_loader, epochs=epochs, criterion=criterion, optimizer=optimizer)
    predictor.calibrate(cal_data_loader, alpha)
    print(predictor.evaluate(test_data_loader))
    

def test_ensemble_predictor():
    ##################################
    # Preparing dataset
    ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fix_randomness(seed=1)
    X, y = build_reg_data(data_name="synthetic")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index1 = int(len(indices) * 0.5)
    part1, part2= np.split(indices, [split_index1])
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[part1, :])
    train_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part1, :])), torch.from_numpy(y[part1]))
    test_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part2, :])), torch.from_numpy(y[part2]))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

    # CP parameters
    epochs = 20
    alpha = 0.1
    
    ##################################
    # Sequential Distribution-free Ensemble Batch Prediction Intervals
    ##################################
    print("########################## EnbPI ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 1, 64, 0.5).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    score_function = split()
    aggregation_function = torch.mean
    predictor = EnsemblePredictor(model, score_function, aggregation_function)
    predictor.fit(train_dataloader=train_data_loader, ensemble_num=5, subset_num=500, epochs=epochs, criterion=criterion, optimizer=optimizer)
    print(predictor.evaluate(test_data_loader, alpha, verbose=True))
    
    ##################################
    # Ensemble Conformal Quantile Regression
    ##################################
    print("########################## EnCQR ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
    quantiles = [alpha / 2, 1 - alpha / 2]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    score_function = CQR()
    aggregation_function = torch.mean
    predictor = EnsemblePredictor(model, score_function, aggregation_function)
    predictor.fit(train_dataloader=train_data_loader, ensemble_num=5, subset_num=500, epochs=epochs, criterion=criterion, optimizer=optimizer)
    print(predictor.evaluate(test_data_loader, alpha, verbose=True))
    

def test_aci_predictor():
    ##################################
    # Preparing dataset
    ##################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fix_randomness(seed=1)
    X, y = build_reg_data(data_name="synthetic")
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index1 = int(len(indices) * 0.5)
    part1, part2= np.split(indices, [split_index1])
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X[part1, :])
    train_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part1, :])), torch.from_numpy(y[part1]))
    test_dataset = TensorDataset(torch.from_numpy(scalerX.transform(X[part2, :])), torch.from_numpy(y[part2]))

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

    # CP parameters
    epochs = 20
    alpha = 0.1
    
    ##################################
    # Adaptive Conformal Inference
    ##################################
    print("########################## ACI ###########################")
    model = build_regression_model("NonLinearNet")(X.shape[1], 2, 64, 0.5).to(device)
    quantiles = [alpha / 2, 1 - alpha / 2]
    criterion = QuantileLoss(quantiles)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    score_function = CQR()
    predictor = ACIPredictor(model, score_function, gamma=0.005)
    predictor.fit(train_dataloader=train_data_loader, alpha=alpha, epochs=epochs, criterion=criterion, optimizer=optimizer)
    print(predictor.evaluate(test_data_loader, verbose=True))
    


if __name__ == "__main__":
    test_split_predictor()
    test_ensemble_predictor()
    test_aci_predictor()