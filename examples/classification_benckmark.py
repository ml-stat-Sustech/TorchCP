# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os
import pickle

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from tqdm import tqdm

from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR, APS, SAPS, RAPS, Margin
from torchcp.classification.utils.metrics import Metrics
from transformers import set_seed
from examples.utils import build_dataset




if __name__ == '__main__':
    # Set the random seed for reproducibility
    set_seed(seed=0)

    #######################################
    # Loading ImageNet dataset and a PyTorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = build_dataset(dataset_name="imagenet", data_mode="test", transform_mode="test")

    # Split the dataset into calibration and test sets
    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1  # Significance level
    predictors = [SplitPredictor]  # List of predictor classes
    score_functions = [THR(), APS(), RAPS(penalty=0.1, kreg=1), SAPS()]  # List of score functions

    # Iterate over each score function and predictor
    for score in score_functions:
        for class_predictor in predictors:
            # Initialize the predictor with the score function and model
            predictor = class_predictor(score, model, temperature=1)
            
            # Calibrate the predictor using the calibration data loader
            predictor.calibrate(cal_data_loader, alpha)
            
            # Print experiment details
            print(
                f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
            
            # Evaluate the predictor using the test data loader and print the results
            print(predictor.evaluate(test_data_loader))
