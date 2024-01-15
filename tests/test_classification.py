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

from torchcp.classification.predictors import SplitPredictor, ClusterPredictor, ClassWisePredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS, Margin
from torchcp.classification.utils.metrics import Metrics
from torchcp.utils import fix_randomness


transform = trn.Compose([trn.Resize(256),
                        trn.CenterCrop(224),
                        trn.ToTensor(),
                        trn.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])

def test_imagenet_logits():
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    fix_randomness(seed=0)
    model_name = 'ResNet101'
    fname = ".cache/" + model_name + ".pkl"
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            dataset = pickle.load(handle)

    else:
        usr_dir = os.path.expanduser('~')
        data_dir = os.path.join(usr_dir, "data")
        dataset = dset.ImageFolder(data_dir + "/imagenet/val",
                                transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=320, shuffle=False, pin_memory=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load model
        model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True).to(device)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in tqdm(data_loader):
                tmp_x, tmp_label = examples[0], examples[1]
                tmp_logits = model(tmp_x)
                logits_list.append(tmp_logits)
                labels_list.append(tmp_label)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        dataset = torch.utils.data.TensorDataset(logits, labels.long())
        with open(fname, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cal_data, val_data = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_logits = torch.stack([sample[0] for sample in cal_data])
    cal_labels = torch.stack([sample[1] for sample in cal_data])

    test_logits = torch.stack([sample[0] for sample in val_data])
    test_labels = torch.stack([sample[1] for sample in val_data])
    
    num_classes = 1000
    
    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1
    predictors = [SplitPredictor, ClassWisePredictor, ClusterPredictor]
    score_functions = [THR(),  APS(), RAPS(1, 0), SAPS(0.2), Margin()]
    for score in score_functions: 
        for class_predictor in predictors:
            predictor = class_predictor(score)
            predictor.calculate_threshold(cal_logits, cal_labels, alpha)
            print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
            prediction_sets = predictor.predict_with_logits(test_logits)

            metrics = Metrics()
            print("Evaluating prediction sets...")
            print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
            print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
            print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
            
            
            

            
def test_imagenet():
    fix_randomness(seed=0)
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data")
    dataset = dset.ImageFolder(data_dir + "/imagenet/val", transform)

    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
        
    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1
    predictors = [SplitPredictor, ClassWisePredictor, ClusterPredictor]
    # score_functions = [THR(),  APS(), RAPS(1, 0), SAPS(0.2), Margin()]
    score_functions = [Margin()]
    for score in score_functions: 
        for class_predictor in predictors:
            predictor = class_predictor(score, model)
            predictor.calibrate(cal_data_loader, alpha)
            print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
            print(predictor.evaluate(test_data_loader))

