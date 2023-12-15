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
from deepcp.classification.utils.metrics import Metrics
from deepcp.utils import fix_randomness

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--predictor', default="Standard", help="Standard | ClassWise | Cluster")
    parser.add_argument('--score', default="THR", help="THR | APS | SAPS")
    parser.add_argument('--penalty', default=1, type=float)
    parser.add_argument('--weight', default=0.2, type=float)
    parser.add_argument('--kreg', default=0, type=int)
    args = parser.parse_args()

    fix_randomness(seed=args.seed)

    model_name = 'ResNet101'
    fname = ".cache/" + model_name + ".pkl"
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            dataset = pickle.load(handle)

    else:
        # load dataset
        transform = trn.Compose([trn.Resize(256),
                                 trn.CenterCrop(224),
                                 trn.ToTensor(),
                                 trn.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])
                                 ])
        usr_dir = os.path.expanduser('~')
        data_dir = os.path.join(usr_dir, "data") 
        dataset = dset.ImageFolder(data_dir + "/imagenet/val",
                                   transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=320, shuffle=False, pin_memory=True)

        # load model
        model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)

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
    if args.score == "THR":
        score_function = THR()
    elif args.score == "APS":
        score_function = APS()
    elif args.score == "RAPS":
        score_function = RAPS(args.penalty,args.kreg)
    elif args.score == "SAPS":
        score_function = SAPS(weight=args.weight)
    alpha = 0.1
    if args.predictor  == "Standard":
        predictor = StandardPredictor(score_function, model=None)
    elif args.predictor  == "ClassWise":   
        predictor = ClassWisePredictor(score_function, model=None)
    elif args.predictor  == "Cluster":   
        predictor = ClusterPredictor(score_function, model=None)
    print(f"The size of calibration set is {cal_labels.shape[0]}.")
    predictor.calculate_threshold(cal_logits, cal_labels, alpha)

    # test examples
    print("Testing examples...")
    prediction_sets = []
    for index, ele in enumerate(test_logits):
        prediction_set = predictor.predict_with_logits(ele)
        prediction_sets.append(prediction_set)

    metrics = Metrics()
    print("Evaluating prediction sets...")
    print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
    print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
    print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
