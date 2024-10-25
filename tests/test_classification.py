# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from tqdm import tqdm
import numpy as np

from torchcp.classification.predictors import SplitPredictor, ClusteredPredictor, ClassWisePredictor
from torchcp.classification.scores import THR, APS, SAPS, RAPS, Margin
from torchcp.classification.utils.metrics import Metrics
from transformers import set_seed
from torchcp.classification.utils import OrdinalClassifier
from torchcp.classification.scores import KNN
from .utils import *

dataset_dir = get_dataset_dir()
model_dir = get_model_dir()


def get_imagenet_logits(model_name):
    fname = f"{dataset_dir}/{model_name}.pkl"
    if os.path.exists(fname):
        with open(fname, 'rb') as handle:
            dataset = pickle.load(handle)

    else:
        dataset = build_dataset(dataset_name = "imagenet", data_mode= "test", transform_mode = "test")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=320, shuffle=False, pin_memory=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # load model
        model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True).to(device)

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
        dataset = torch.utils.data.TensorDataset(logits, labels.long())
        with open(fname, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cal_data, val_data = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_logits = torch.stack([sample[0] for sample in cal_data])
    cal_labels = torch.stack([sample[1] for sample in cal_data])

    test_logits = torch.stack([sample[0] for sample in val_data])
    test_labels = torch.stack([sample[1] for sample in val_data])
    num_classes = 1000
    return cal_logits, cal_labels, test_logits, test_labels, num_classes


def test_imagenet_logits():
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    set_seed(seed=0)
    model_name = 'ResNet101'
    cal_logits, cal_labels, test_logits, test_labels, num_classes = get_imagenet_logits(model_name)

    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1
    predictors = [SplitPredictor, ClassWisePredictor, ClusteredPredictor]
    score_functions = [THR(), APS(), RAPS(1, 0), SAPS(0.2), Margin()]
    for score in score_functions:
        for the_predictor in predictors:
            predictor = the_predictor(score)
            predictor.calculate_threshold(cal_logits, cal_labels, alpha)
            print(
                f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
            prediction_sets = predictor.predict_with_logits(test_logits)

            metrics = Metrics()
            print("Evaluating prediction sets...")
            print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
            print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
            print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
            print(f"VioClasses: {metrics('VioClasses')(prediction_sets, test_labels, alpha, num_classes)}.")
            print(f"DiffViolation: {metrics('DiffViolation')(test_logits, prediction_sets, test_labels, alpha)}.")


def test_imagenet_logits_unrandomized():
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    set_seed(seed=0)
    model_name = 'ResNet101'
    cal_logits, cal_labels, test_logits, test_labels, num_classes = get_imagenet_logits(model_name)

    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1
    score_functions = [APS(randomized=False), RAPS(1, 0,randomized=False), SAPS(0.2,randomized=False)]
    for score in score_functions:
        predictor = SplitPredictor(score)
        predictor.calculate_threshold(cal_logits, cal_labels, alpha)
        print(
            f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
        prediction_sets = predictor.predict_with_logits(test_logits)

        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
        print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
        print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
        print(f"VioClasses: {metrics('VioClasses')(prediction_sets, test_labels, alpha, num_classes)}.")
        print(f"DiffViolation: {metrics('DiffViolation')(test_logits, prediction_sets, test_labels, alpha)}.")
        
        
        
def test_imagenet():
    set_seed(seed=0)
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    model_name = 'ResNet101'
    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = build_dataset(dataset_name = "imagenet", data_mode= "test", transform_mode = "test")

    cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [25000, 25000])
    cal_data_loader = torch.utils.data.DataLoader(cal_dataset, batch_size=1024, shuffle=False, num_workers=4,
                                                  pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4,
                                                   pin_memory=True)

    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1
    predictors = [SplitPredictor, ClassWisePredictor, ClusteredPredictor]
    # score_functions = [THR(),  APS(), RAPS(1, 0), SAPS(0.2), Margin()]
    score_functions = [APS()]
    for score in score_functions:
        for class_predictor in predictors:
            predictor = class_predictor(score, model, temperature=1)
            predictor.calibrate(cal_data_loader, alpha)
            print(
                f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
            print(predictor.evaluate(test_data_loader))


def test_calibration():
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    set_seed(seed=0)

    model_name = 'ResNet101'
    cal_logits, cal_labels, test_logits, test_labels, num_classes = get_imagenet_logits(model_name)

    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1
    predictors = [SplitPredictor, ClassWisePredictor, ClusteredPredictor]
    score = SAPS(0.2)
    temperatures = [0.5, 1, 1.5]
    for temperature in temperatures:
        for class_predictor in predictors:
            predictor = class_predictor(score, temperature=temperature)
            predictor.calculate_threshold(cal_logits, cal_labels, alpha)
            print(
                f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}, Temperature : {temperature}")
            prediction_sets = predictor.predict_with_logits(test_logits)

            metrics = Metrics()
            print("Evaluating prediction sets...")
            print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
            print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
            print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")


def test_imagenet_logits_types():
    #######################################
    # Loading ImageNet dataset and a pytorch model
    #######################################
    set_seed(seed=0)
    model_name = 'ResNet101'
    cal_logits, cal_labels, test_logits, test_labels, num_classes = get_imagenet_logits(model_name)

    #######################################
    # A standard process of conformal prediction
    #######################################
    alpha = 0.1

    tranformation_types = ["identity", "softmax", "log_softmax", "log"]
    for tranformation_type in tranformation_types:
        score_function = APS(score_type=tranformation_type)
        predictor = SplitPredictor(score_function)
        predictor.calculate_threshold(cal_logits, cal_labels, alpha)
        print(
            f"Experiment--Data : ImageNet, Model : {model_name}, Score : {score_function.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}, Score_type: {tranformation_type}")
        prediction_sets = predictor.predict_with_logits(test_logits)

        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
        print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
        print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
        print(f"VioClasses: {metrics('VioClasses')(prediction_sets, test_labels, alpha, num_classes)}.")
        print(f"DiffViolation: {metrics('DiffViolation')(test_logits, prediction_sets, test_labels, alpha)}.")


def test_ordinal_classification():
    set_seed(seed=0)

    num_classes = 10
    num_per_class = 2000
    cov_scale = 1.2
    means = [[i, i] for i in range(num_classes)]
    covs = [np.random.rand(2, 2) * cov_scale for i in range(num_classes)]

    labels = np.array([item for sublist in [[i] * num_per_class for i in range(num_classes)] for item in sublist])

    x = np.array([np.random.multivariate_normal(means[i], covs[i], num_per_class) for i in range(num_classes)])
    x = x.reshape((num_classes * num_per_class, 2))

    shuffled_indices = [i for i in range(num_classes * num_per_class)]
    random.shuffle(shuffled_indices)
    x = x[shuffled_indices, :]

    labels = labels[shuffled_indices]

    train_ratio = 0.2
    train_num = int(train_ratio * num_classes * num_per_class)

    x_tr = x[:train_num, :]
    labels_tr = labels[:train_num]
    x_rest = x[train_num:, :]
    labels_rest = labels[train_num:]

    class Data(Dataset):
        def __init__(self, x_train, y_train):
            self.x = torch.from_numpy(x_train)
            self.y = torch.from_numpy(y_train)
            self.len = self.x.shape[0]

        def __getitem__(self, index):
            return self.x[index], self.y[index]

        def __len__(self):
            return self.len

    train_dataset = Data(x_tr, labels_tr)
    trainloader = DataLoader(dataset=train_dataset, batch_size=64)

    class MultiClassModel(nn.Module):
        def __init__(self, D_in, H, D_out):
            super(MultiClassModel, self).__init__()
            self.linear1 = nn.Linear(D_in, H)
            self.linear2 = nn.Linear(H, D_out)

        def forward(self, x):
            x = torch.sigmoid(self.linear1(x.float()))
            x = self.linear2(x)
            return x

    labels_rest = torch.from_numpy(labels_rest).cuda()
    mask = torch.rand_like(labels_rest, dtype=torch.float32) > 0.5
    cal_labels = labels_rest[mask]
    test_labels = labels_rest[~mask]

    for phi in ["abs", "square"]:
        print(f"Experiment--Data : ImageNet, phi : {phi}.")
        base_model = MultiClassModel(2, 50, num_classes)
        model = OrdinalClassifier(base_model, phi)
        model.cuda()

        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.05
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        n_epochs = 100
        for epoch in range(n_epochs):
            for x, y in trainloader:
                optimizer.zero_grad()
                z = model(x.cuda())
                loss = criterion(z, y.cuda())
                loss.backward()
                optimizer.step()
            if epoch % 100 == 0:
                print('epoch {}, loss {}'.format(epoch, loss.item()))

        logits_rest = model(torch.from_numpy(x_rest).cuda())
        cal_logits = logits_rest[mask, :]
        test_logits = logits_rest[~mask]

        alpha = 0.1

        predictor = SplitPredictor(APS())
        predictor.calculate_threshold(cal_logits, cal_labels, alpha)
        prediction_sets = predictor.predict_with_logits(test_logits)
        metrics = Metrics()
        print("Evaluating prediction sets...")
        print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
        print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
        print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
        print(f"VioClasses: {metrics('VioClasses')(prediction_sets, test_labels, alpha, num_classes)}.")
        print(f"DiffViolation: {metrics('DiffViolation')(test_logits, prediction_sets, test_labels, alpha)}.")


def test_KNN_Score():
    set_seed(seed=0)
    
    
    train_dataset = build_dataset("cifar10", "train", "train")
    
    

    num_classes = 10
    num_epochs = 30
    batch_size = 1024
    lr = 0.001

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = torchvision.models.resnet101(weights="IMAGENET1K_V1", progress=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.cuda()

    model_pkl_path = os.path.join(model_dir,"resnet101_cifar10.pth")
    if os.path.exists(model_pkl_path):
        pretrained_dict = torch.load(model_pkl_path)
        model.load_state_dict(pretrained_dict)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            pre_acc = 0
            for inputs, labels in train_dataloader:
                labels = labels.cuda()
                inputs = inputs.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)

                pre_acc += torch.sum(torch.argmax(outputs, axis=1) == labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader)},ACC.: {pre_acc / len(train_dataset)}")

        torch.save(model.state_dict(), model_pkl_path)

    model.eval()
    def get_features(dataloader, the_model):
        the_features = []

        def hook_fn(module, input, output):
            the_features.append(output.clone())

        layer = the_model.avgpool
        layer.register_forward_hook(hook_fn)
        outputsize = 2048
        with torch.no_grad():
            for x, targets in tqdm(dataloader):
                batch_logits = the_model(x.cuda())
        features = torch.reshape(torch.cat(the_features, dim=0), (-1, outputsize)).detach()
        return features
    
    
    train_dataset = build_dataset("cifar10", data_mode="train", transform_mode="test")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_labels = torch.stack([torch.tensor(sample[1]) for sample in train_dataset])
    
    test_dataset = build_dataset("cifar10", data_mode="test", transform_mode="test")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_labels = torch.stack([torch.tensor(sample[1]) for sample in test_dataset])
        
    train_features = get_features(train_dataloader, model)
    test_features = get_features(test_dataloader, model)

    
    score = KNN(train_features, train_labels, num_classes, k=2, p=2)

    mask = torch.rand_like(test_labels, dtype=torch.float32) > 0.5
    cal_features = test_features[mask]
    test_features = test_features[~mask]
    cal_labels = test_labels[mask]
    test_labels = test_labels[~mask]

    ############################################
    # A standard process of conformal prediction
    ############################################
    alpha = 0.1

    predictor = SplitPredictor(score)
    predictor.calculate_threshold(cal_features, cal_labels, alpha)
    print(
        f"Experiment--Data : ImageNet, Model : ResNet101, Score : {score.__class__.__name__}, Predictor : {predictor.__class__.__name__}, Alpha : {alpha}")
    prediction_sets = predictor.predict_with_logits(test_features)

    metrics = Metrics()
    print("Evaluating prediction sets...")
    print(f"Coverage_rate: {metrics('coverage_rate')(prediction_sets, test_labels)}.")
    print(f"Average_size: {metrics('average_size')(prediction_sets, test_labels)}.")
    print(f"CovGap: {metrics('CovGap')(prediction_sets, test_labels, alpha, num_classes)}.")
    print(f"VioClasses: {metrics('VioClasses')(prediction_sets, test_labels, alpha, num_classes)}.")
