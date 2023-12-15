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
import torch.optim as optim
from torch.nn.functional import softmax
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from deepcp.classification.predictor import StandardPredictor, ClusterPredictor, ClassWisePredictor, WeightedPredictor
from deepcp.classification.scores import THR, APS, SAPS,RAPS
from deepcp.classification.loss_function import ConfTr
from deepcp.classification.utils.metrics import Metrics
from deepcp.utils import fix_randomness
from dataset import build_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Covariate shift')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--predictor', default="Standard", help="Standard")
    parser.add_argument('--score', default="THR", help="THR")
    parser.add_argument('--loss', default="CE", help="CE | ConfTr")
    args = parser.parse_args()
    res = {'Coverage_rate': 0, 'Average_size': 0}
    num_trials = 1
    for seed in range(num_trials):
        fix_randomness(seed=seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ##################################
        # Invalid prediction sets
        ##################################
        train_dataset = build_dataset("mnist")
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)    
        
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(28*28, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = x.view(-1, 28*28)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = Net().to(device)
        
        if args.loss == "CE":
            criterion = nn.CrossEntropyLoss()
        elif args.loss == "ConfTr":
            predictor =  StandardPredictor(score_function = THR(score_type= "log_softmax"))
            criterion = ConfTr(weights=0.01,
                               predictor = predictor, 
                               alpha=0.05,
                               device = device,             
                               fraction=0.5,
                               loss_types = "valid",
                               base_loss_fn = nn.CrossEntropyLoss())
        else:
            raise NotImplementedError
            
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        def train(model, device, train_loader, optimizer, epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        checkpoint_path = f'.cache/conformal_training_model_checkpoint_{args.loss}_seed={seed}.pth'
        # if os.path.exists(checkpoint_path):
        #     checkpoint = torch.load(checkpoint_path)
        #     model.load_state_dict(checkpoint['model_state_dict'])
        # else:
        for epoch in range(1, 10):
            train(model, device, train_data_loader, optimizer, epoch)
        
        torch.save({'model_state_dict': model.state_dict(),}, checkpoint_path)
        
        test_dataset = build_dataset("mnist", mode= 'test')
        cal_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [5000, 5000])
        cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1600, shuffle=False, pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1600, shuffle=False, pin_memory=True)
        
        

        if args.score == "THR":
            score_function = THR()
        elif args.score == "APS":
            score_function = APS()
        elif args.score == "RAPS":
            score_function = RAPS(args.penalty, args.kreg)
        elif args.score == "SAPS":
            score_function = SAPS(weight= args.weight)
            
        alpha = 0.01
        if args.predictor  == "Standard":
            predictor = StandardPredictor(score_function, model)
        elif args.predictor  == "ClassWise":   
            predictor = ClassWisePredictor(score_function, model)
        elif args.predictor  == "Cluster":   
            predictor = ClusterPredictor(score_function, model, args.seed)
        predictor.calibrate(cal_data_loader,  alpha)

        # test examples
        tmp_res = predictor.evaluate(test_data_loader)
        res['Coverage_rate'] += tmp_res['Coverage_rate']/num_trials
        res['Average_size'] += tmp_res['Average_size']/num_trials
        

    print(res)



    
    
    





