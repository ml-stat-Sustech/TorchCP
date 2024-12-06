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
from transformers import set_seed

from examples.utils import build_dataset
from torchcp.classification.loss import ConfTr, CDLoss, ConfTS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import THR


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


def train(model, device, train_loader, criterion, optimizer, epoch, use_conf=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if use_conf:
            loss = criterion(output, target) + F.cross_entropy(output, target)
        else:
            loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()


def run_experiment(model, device, train_loader, cal_loader, test_loader, criterion,
                   optimizer, epochs, alpha, use_conf=False):
    print(f"Starting to train the model with {'conformal loss' if use_conf else 'cross entropy'} ...")

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch, use_conf)

    score_function = THR()
    predictor = SplitPredictor(score_function, model)
    predictor.calibrate(cal_loader, alpha)
    result = predictor.evaluate(test_loader)

    return result


def setup_data_and_model(device, batch_size=512):
    train_dataset = build_dataset("mnist")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    test_dataset = build_dataset("mnist", data_mode='test')
    cal_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [5000, 5000])

    cal_loader = torch.utils.data.DataLoader(
        cal_dataset, batch_size=1600, shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1600, shuffle=False, pin_memory=True)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    return train_loader, cal_loader, test_loader, model, optimizer


if __name__ == '__main__':
    alpha = 0.1
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed=0)

    losses = {
        "CE": None,  # Will use standard CrossEntropyLoss in training loop
        "ConfTr": lambda: ConfTr(
            weight=0.01,
            predictor=SplitPredictor(score_function=THR(score_type="log_softmax")),
            alpha=0.1,
            fraction=0.5,
            loss_type="valid"
        ),
        "CDLoss": lambda: CDLoss(
            weight=0.01,
            predictor=SplitPredictor(score_function=THR(score_type="log_softmax")),
        )
    }

    results = {}

    for loss_name, loss_fn in losses.items():
        print(f"\n{'=' * 20} {loss_name} {'=' * 20}")

        # Setup fresh data and model for each experiment
        train_loader, cal_loader, test_loader, model, optimizer = setup_data_and_model(device)

        # Run experiment
        result = run_experiment(
            model, device, train_loader, cal_loader, test_loader,
            loss_fn() if loss_fn else nn.CrossEntropyLoss(),
            optimizer, epochs, alpha,
            use_conf=(loss_name != "CE")
        )

        results[loss_name] = result
        print(f"Result--Coverage_rate: {result['Coverage_rate']:.4f}, "
              f"Average_size: {result['Average_size']:.4f}")

    # Print comparative results
    print("\nComparative Results:")
    print("-" * 60)
    print(f"{'Method':<10} {'Coverage Rate':<15} {'Average Size':<15}")
    print("-" * 60)
    for method, result in results.items():
        print(f"{method:<10} {result['Coverage_rate']:.4f}{'':8} {result['Average_size']:.4f}")
