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


from examples.utils import get_dataset_dir
set_seed(seed=0)

#######################################
#Preparing a calibration data and a test data
#######################################
transform = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761])
])
dataset =  torchvision.datasets.CIFAR100(
            root=get_dataset_dir(),
            train=False,
            download=True,
            transform=transform
        )
cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [5000, 5000])
cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

#######################################
# Preparing a pytorch model
#######################################
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(model_device)
model.eval()


#######################################
# A standard process of conformal prediction
#######################################
alpha = 0.1  # Significance level
predictor = SplitPredictor(score_function=THR(), model=model)
predictor.calibrate(cal_dataloader, alpha=0.1)

test_instances, test_labels = test_dataset[0]
predict_sets = predictor.predict(test_instances.unsqueeze(0))
print(predict_sets)


#########################################
# Evaluating the coverage rate and average set size on a given dataset.
########################################
result_dict = predictor.evaluate(test_dataloader)
print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
print(f"Average Set Size: {result_dict['average_size']:.4f}")
