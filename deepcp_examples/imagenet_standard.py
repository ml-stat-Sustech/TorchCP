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
from torch.nn.functional import softmax
from tqdm import tqdm

from deepcp.classification.predictor import StandardPredictor,ClusterPredictor,ClassWisePredictor
from deepcp.classification.scores import THR, APS, SAPS,RAPS
from deepcp.classification.utils.metircs import Metrics
from deepcp.utils import fix_randomness

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int )
    parser.add_argument('--predictor', default= "Standard", help= "Standard | ClassWise | Cluster" )
    parser.add_argument('--score', default="THR", help= "THR | APS | SAPS" )
    parser.add_argument('--penalty', default=1, type=float )
    parser.add_argument('--weight', default=1, type=float )
    parser.add_argument('--kreg', default=0, type=int )
    args = parser.parse_args()

    fix_randomness(seed=args.seed)

    model_name = 'ResNet101'
    # load model
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    model_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model.to(model_device)


   
    # load dataset
    transform = trn.Compose([trn.Resize(256),
                                trn.CenterCrop(224),
                                trn.ToTensor(),
                                trn.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                                ])
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data") 
    dataset = dset.ImageFolder(data_dir + "/imagenet/val", transform)
    
    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1600, shuffle=False, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1600, shuffle=False, pin_memory=True)
    
    

    num_classes = 1000
    if args.score == "THR":
        score_function = THR()
    elif args.score == "APS":
        score_function = APS()
    elif args.score == "RAPS":
        score_function = RAPS(args.penalty, args.kreg)
    elif args.score == "SAPS":
        score_function = SAPS(weight= args.weight)
    alpha = 0.1
    if args.predictor  == "Standard":
        predictor = StandardPredictor(score_function, model)
    elif args.predictor  == "ClassWise":   
        predictor = ClassWisePredictor(score_function, model)
    elif args.predictor  == "Cluster":   
        predictor = ClusterPredictor(score_function, model, args.seed)
    print(f"The size of calibration set is {len(cal_dataset)}.")
    predictor.calibrate(cal_data_loader,  alpha)

    # test examples
    print("Testing examples...")
    prediction_sets = []
    labels_list = []
    with torch.no_grad():
            for  examples in tqdm(test_data_loader):
                tmp_x, tmp_label = examples[0], examples[1]            
                prediction_sets_batch = predictor.predict(tmp_x)
                prediction_sets.extend(prediction_sets_batch)
                labels_list.append(tmp_label)
    test_labels = torch.cat(labels_list)
    
    metrics = Metrics()
    print("Etestuating prediction sets...")
    print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
    print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
    print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
