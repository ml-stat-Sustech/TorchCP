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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .utils import build_dataset
from torchcp.classification.loss import ConfTr
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import THR
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


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    alpha = 0.01
    losses = ["CE", "ConfTr"]
    for loss in losses:
        result = {}
        print(f"############################## {loss} #########################")

        predictor = SplitPredictor(score_function=THR(score_type="log_softmax"))
        if loss == "CE":
            criterion = nn.CrossEntropyLoss()
        elif loss == "ConfTr":
            criterion = ConfTr(weight=0.01,
                               predictor=predictor,
                               alpha=0.1,
                               fraction=0.5,
                               loss_type="valid",
                               base_loss_fn=nn.CrossEntropyLoss())
        else:
            NotImplementedError

        fix_randomness(seed=0)
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

        print(f"Starting to train the model ...")
        for epoch in range(1, 10):
            train(model, device, train_data_loader, criterion, optimizer, epoch)

        score_function = THR()

        predictor = SplitPredictor(score_function, model)
        predictor.calibrate(cal_data_loader, alpha)
        result = predictor.evaluate(test_data_loader)
        print(f"Result--Coverage_rate: {result['Coverage_rate']}, Average_size: {result['Average_size']}")
