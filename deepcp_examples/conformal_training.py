# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# @Time : 13/12/2023  21:13


# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import os
import json

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.nn.functional import softmax
import torch.nn as nn
from tqdm import tqdm

from deepcp.classification.predictor import StandardPredictor, ClusterPredictor, ClassWisePredictor, WeightedPredictor
from deepcp.classification.scores import THR, APS, SAPS, RAPS
from deepcp.classification.utils.metircs import Metrics
from deepcp.utils import fix_randomness
from dataset import build_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Covariate shift')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--predictor', default="Standard", help="Standard")
    parser.add_argument('--score', default="APS", help="THR")
    args = parser.parse_args()
    fix_randomness(seed=args.seed)


    ##################################
    # Invalid prediction sets
    ##################################
    train_dataset = build_dataset("minst")
    test_dataset = build_dataset("minist", mode = "test")





