# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import set_seed

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from examples.utils import get_dataset_dir
from torchcp.classification.score import APS
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.trainer import UncertaintyAwareTrainer


def setup_data_and_model(device):
    ########################################
    # Prepare train dataset
    ########################################
    augmentation = [
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale = (0.7, 0.7), ratio = (0.3, 3.3), value=0),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    transform = transforms.Compose(augmentation)
    train_dataset = torchvision.datasets.CIFAR10(
        root=get_dataset_dir(), 
        train=True, 
        download=True, 
        transform=transform)

    random_indices = torch.randperm(len(train_dataset))
    random_indices_tr = random_indices[:45000]
    random_indices_va = random_indices[45000:]

    trainset_sample = torch.utils.data.Subset(train_dataset, random_indices_tr)
    valset_sample = torch.utils.data.Subset(train_dataset, random_indices_va)

    train_loader = torch.utils.data.DataLoader(trainset_sample, batch_size=750, shuffle=True, num_workers=2, drop_last=True)
    val_loader = torch.utils.data.DataLoader(valset_sample, batch_size=750, shuffle=False, num_workers=2)

    ########################################
    # Prepare test dataset
    ########################################

    augmentation = [
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor((0.4914, 0.4822, 0.4465)), torch.Tensor((0.2023, 0.1994, 0.2010)))
    ]
    transform = transforms.Compose(augmentation)
    test_dataset = torchvision.datasets.CIFAR10(
        root=get_dataset_dir(), 
        train=False, 
        download=True, 
        transform=transform)
    
    random_indices = torch.randperm(len(test_dataset))
    random_indices_ca = random_indices[:int(len(test_dataset) * 0.5)]
    random_indices_te = random_indices[int(len(test_dataset) * 0.5):]

    calset_sample = torch.utils.data.Subset(test_dataset, random_indices_ca)
    testset_sample = torch.utils.data.Subset(test_dataset, random_indices_te)

    cal_loader = torch.utils.data.DataLoader(calset_sample, batch_size=750, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset_sample, batch_size=750, shuffle=False, num_workers=2)

    model = models.resnet18(weights=None).to(device)
    model.fc = nn.Linear(model.fc.in_features, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return train_loader, val_loader, cal_loader, test_loader, model, optimizer


if __name__ == '__main__':
    set_seed(seed=42)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    #######################################
    # Loading dataset, a model and Conformal Learning Trainer
    #######################################
    train_loader, val_loader, cal_loader, test_loader, model, optimizer = setup_data_and_model(device)
    ua_trainer = UncertaintyAwareTrainer(model, device=device)
    
    #######################################
    # Conformal Learning
    #######################################
    ua_trainer.train(train_loader, val_loader, num_epochs=10)

    ########################################
    # Split Conformal prediction
    ########################################
    predictor = SplitPredictor(score_function=APS(), model=ua_trainer.model)
    predictor.calibrate(cal_loader, alpha=0.1)
    
    x_list = []
    y_list = []
    for tmp_x, tmp_y in test_loader:
        x_list.append(tmp_x)
        y_list.append(tmp_y)
    X_data = torch.cat(x_list)
    Y_data = torch.cat(y_list)

    pred_set = predictor.predict(X_data[0:1])
    print(pred_set)

    result_dict = predictor.evaluate(test_loader)
    print(f"Marginal Coverage: {result_dict['coverage_rate']:.4f}")
    print(f"Average Set Size: {result_dict['average_size']:.4f}")