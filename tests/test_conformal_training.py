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
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import build_dataset
from torchcp.classification.loss import ConfTr
from torchcp.classification.predictors import SplitPredictor, ClusteredPredictor, ClassWisePredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS
from torchcp.utils import fix_randomness



class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(28 * 28, 500)
                self.fc2 = nn.Linear(500, 10)

            def forward(self, x):
                x = x.view(-1, 28 * 28)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
            
def train(model, device, train_loader,criterion,  optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
            

def test_training():
    alpha = 0.01
    num_trials = 5
    result = {}
    for loss in ["CE", "ConfTr"]:
        print(f"############################## {loss} #########################")
        result[loss] = {}
        if loss == "CE":
            criterion = nn.CrossEntropyLoss()
        elif loss == "ConfTr":
            predictor = SplitPredictor(score_function=THR(score_type="log_softmax"))
            criterion = ConfTr(weight=0.01,
                        predictor=predictor,
                        alpha=0.05,
                        fraction=0.5,
                        loss_type="valid",
                        base_loss_fn=nn.CrossEntropyLoss())
        else:
            raise NotImplementedError
        for seed in range(num_trials):
            fix_randomness(seed=seed)
            ##################################
            # Training a pytorch model
            ##################################
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_dataset = build_dataset("mnist")
            train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
            test_dataset = build_dataset("mnist", mode='test')
            cal_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [5000, 5000])
            cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1600, shuffle=False, pin_memory=True)
            test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1600, shuffle=False, pin_memory=True)
            
            model = Net().to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
            for epoch in range(1, 10):
                train(model, device, train_data_loader, criterion, optimizer, epoch)
                
            for score in ["THR", "APS", "RAPS", "SAPS"]:
                if score == "THR":
                    score_function = THR()
                elif score == "APS":
                    score_function = APS()
                elif score == "RAPS":
                    score_function = RAPS(1, 0)
                elif score == "SAPS":
                    score_function = SAPS(weight=0.2)
                if score not in result[loss]:
                    result[loss][score] = {}
                    result[loss][score]['Coverage_rate'] = 0
                    result[loss][score]['Average_size'] = 0
                predictor = SplitPredictor(score_function, model)
                predictor.calibrate(cal_data_loader, alpha)                
                tmp_res = predictor.evaluate(test_data_loader)
                result[loss][score]['Coverage_rate'] += tmp_res['Coverage_rate'] / num_trials
                result[loss][score]['Average_size'] += tmp_res['Average_size'] / num_trials
                
        for score in ["THR", "APS", "RAPS", "SAPS"]:
            print(f"Score: {score}. Result is {result[loss][score]}")
