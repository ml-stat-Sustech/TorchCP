# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from transformers import set_seed
from tqdm import tqdm
import pickle

from examples.utils import build_dataset, get_others_dir
from torchcp.classification import Metrics
from torchcp.classification.predictor import ClusteredPredictor, ClassWisePredictor, SplitPredictor
from torchcp.classification.score import THR, APS, SAPS, RAPS, Margin, TOPK

transform = trn.Compose([trn.Resize(256),
                        trn.CenterCrop(224),
                        trn.ToTensor(),
                        trn.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                        ])


def load_cached_logits(model_name, model):
    fname = os.path.join(get_others_dir(), f"{model_name}_imagenet_logits.pkl")
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            return pickle.load(handle)
    else:
        dataset = build_dataset('imagenet', data_mode='test',transform_mode="test")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=320, shuffle=False, pin_memory=True)
        device = next(model.parameters()).device
    
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in tqdm(data_loader):
                tmp_x, tmp_label = examples[0].to(device), examples[1].to(device)
                tmp_logits = model(tmp_x)
                logits_list.append(tmp_logits)
                labels_list.append(tmp_label)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        processed_dataset = torch.utils.data.TensorDataset(logits, labels.long())
        with open(fname, 'wb') as handle:
            pickle.dump(processed_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return processed_dataset

def run_experiment(score_function, model_name, alpha, cal_logits,cal_labels, test_logits, test_labels, randomized):
    
    predictor = SplitPredictor(score_function, model)
    print(f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}, Randomized : {randomized}")
    
    predictor.calculate_threshold(cal_logits, cal_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    metrics = Metrics()
    print("Evaluating prediction sets...")
    print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
    print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
    print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
    print("-" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)
    args = parser.parse_args()

    set_seed(seed=args.seed)
    
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(model_device)
    model.eval()
    
    num_classes = 1000
    
    
    dataset = load_cached_logits(model_name, model)
    cal_data, val_data = torch.utils.data.random_split(dataset, [20000, 30000])
    
    # Extract logits and labels
    cal_logits = torch.stack([sample[0] for sample in cal_data])
    cal_labels = torch.stack([sample[1] for sample in cal_data])
    test_logits = torch.stack([sample[0] for sample in val_data])
    test_labels = torch.stack([sample[1] for sample in val_data])

    #######################################
    # A standard process of conformal prediction
    #######################################    
    scoring_methods = [THR(), APS(), SAPS(), RAPS(penalty=0), Margin(), TOPK()]
    
    for score_function in scoring_methods:
        run_experiment(
            score_function=score_function,
            model_name=model_name,
            alpha=args.alpha,
            cal_logits=cal_logits,
            cal_labels=cal_labels,
            test_logits=test_logits,
            test_labels=test_labels,
            randomized = True
        )
        
    scoring_methods = [APS(randomized=False), SAPS(randomized=False), RAPS(randomized=False)]
    
    for score_function in scoring_methods:
        run_experiment(
            score_function=score_function,
            model_name=model_name,
            alpha=args.alpha,
            cal_logits=cal_logits,
            cal_labels=cal_labels,
            test_logits=test_logits,
            test_labels=test_labels,
            randomized = False
        )