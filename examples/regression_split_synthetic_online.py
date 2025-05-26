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

from examples.regression_cqr_synthetic import prepare_dataset
from torchcp.regression.predictor import SplitPredictor
from torchcp.regression.score import ABS
from torchcp.regression.utils import build_regression_model


if __name__ == "__main__":
    # get dataloader
    train_loader, cal_loader, test_loader = prepare_dataset(train_ratio=0.4, cal_ratio=0.2, batch_size=128)
    # build regression model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_regression_model("NonLinearNet")(next(iter(train_loader))[0].shape[1], 2, 64, 0.5).to(device)

    # CP
    alpha = 0.1  # confidence level
    predictor = SplitPredictor(score_function=ABS(), model=None)

    # Step0 (optional): train regression model
    ## We've provided an auxiliary function here to help with model training, 
    ## or user can simply enter the trained model in predictor and ignore this code
    predictor.train(train_loader, alpha=alpha, epochs=100, lr=0.01, device=device, verbose=True)

    cover_count = 0
    total = 0
    set_size_sum = 0

    cal_dataset = cal_loader.dataset
    test_dataset = test_loader.dataset
    for i in tqdm(range(len(test_dataset))):
        # calibration
        predictor.calibrate(cal_loader, alpha=alpha)
        # prediction
        x, y = test_dataset[i]
        x = x.to(device)
        y = y.to(device)
        prediction_intervals = predictor.predict(x)[0][0]
        covered = 1 if prediction_intervals[0] <= y <= prediction_intervals[1] else 0
        cover_count += int(covered)
        set_size_sum += prediction_intervals.sum()
        total += 1
        
        cal_dataset = torch.utils.data.ConcatDataset([cal_dataset, TensorDataset(x.unsqueeze(0), y.unsqueeze(0))])

    coverage_rate = cover_count / total
    average_set_size = set_size_sum / total

    print(f"Online CP Coverage Rate: {coverage_rate:.4f}")
    print(f"Online CP Average Set Size: {average_set_size:.4f}")