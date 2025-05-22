# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, SAGEConv


def get_dataset_dir():
    dataset_dir = os.path.join(
        os.path.expanduser('~'), '.cache/torchcp/datasets')
    path = Path(dataset_dir)
    path.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def get_model_dir():
    dataset_dir = os.path.join(os.path.expanduser('~'), '.cache/torchcp/hub')
    path = Path(dataset_dir)
    path.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def get_others_dir():
    dataset_dir = os.path.join(os.path.expanduser('~'), '.cache/torchcp/others')
    path = Path(dataset_dir)
    path.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def download_github(url, save_path):
    raw_url = url.replace(
        "github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    response = requests.get(raw_url)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Files already downloaded: {save_path}")
    else:
        print(f"Download failed: {response.status_code}")


def build_reg_data(data_name="community"):
    if data_name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        dataset_dir = get_dataset_dir()
        attrib_path = os.path.join(dataset_dir, 'communities_attributes.csv')
        dataset_path = os.path.join(dataset_dir, 'communities.data')
        if not os.path.exists(attrib_path):
            attrib_github_url = "https://github.com/vbordalo/Communities-Crime/blob/master/attributes.csv"

            download_github(attrib_github_url, attrib_path)
            attrib_github_url = "https://github.com/vbordalo/Communities-Crime/blob/master/communities.data"
            download_github(attrib_github_url, dataset_path)

        attrib = pd.read_csv(attrib_path, delim_whitespace=True)
        data = pd.read_csv(dataset_path, names=attrib['attributes'])

        data = data.drop(columns=['state', 'county',
                                  'community', 'communityname',
                                  'fold'], axis=1)
        data = data.replace('?', np.nan)

        # Impute mean values for samples with missing values

        # imputer = SimpleImputer(missing_values = 'NaN', strategy = 'mean')

        # imputer = imputer.fit(data[['OtherPerCap']])
        # data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data['OtherPerCap'] = data['OtherPerCap'].astype("float")
        mean_value = data['OtherPerCap'].mean()
        data['OtherPerCap'].fillna(value=mean_value, inplace=True)
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

    elif data_name == "synthetic":

        n = 10000
        X = np.random.rand(n, 5)
        y_wo_noise = 10 * np.sin(X[:, 0] * X[:, 1] * np.pi) + \
                     20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
        eplison = np.zeros(n)
        phi = theta = 0.8
        delta_t_1 = np.random.randn()
        for i in range(1, n):
            delta_t = np.random.randn()
            eplison[i] = phi * eplison[i - 1] + delta_t_1 + theta * delta_t
            delta_t_1 = delta_t

        y = y_wo_noise + eplison

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y


class NonLinearNet(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_size, dropout):
        super(NonLinearNet, self).__init__()
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.dropout = dropout
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.out_shape),
        )

    def forward(self, x):
        return self.base_model(x)


class Softmax(nn.Module):
    def __init__(self, in_shape, out_shape, hidden_size, dropout):
        super(Softmax, self).__init__()
        self.base_model = nn.Sequential(
            NonLinearNet(in_shape, out_shape, hidden_size, dropout),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.base_model(x)


def build_regression_model(model_name="NonLinearNet"):
    if model_name == "NonLinearNet":
        return NonLinearNet
    elif model_name == "NonLinearNet_with_Softmax":
        return Softmax
    else:
        raise NotImplementedError


def build_dataset(dataset_name, data_mode="train", transform_mode="train"):
    if dataset_name == 'imagenet':
        usr_dir = os.path.expanduser('~')
        dataset_dir = os.path.join(usr_dir, "data")
        if transform_mode == "test":
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])
        else:
            raise NotImplementedError

        if data_mode == "test":
            dataset = dset.ImageFolder(
                dataset_dir + "/datasets/imagenet/val", transform)
        else:
            raise NotImplementedError


    elif dataset_name == 'imagenetv2':
        if transform == None:
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])

        dataset = ImageNetV2Dataset(os.path.join(dataset_dir, "imagenetv2/imagenetv2-matched-frequency-format-val"),
                                    transform)

    elif dataset_name == 'mnist':
        dataset_dir = get_dataset_dir()

        transform = trn.Compose([
            trn.ToTensor(),
            trn.Normalize((0.1307,), (0.3081,))
        ])
        if data_mode == "train":
            dataset = dset.MNIST(dataset_dir, train=True,
                                 download=True, transform=transform)
        elif data_mode == "test":
            dataset = dset.MNIST(dataset_dir, train=False,
                                 download=True, transform=transform)

    elif dataset_name == 'cifar10':
        dataset_dir = get_dataset_dir()

        mean = (0.492, 0.482, 0.446)
        std = (0.247, 0.244, 0.262)

        if transform_mode == "train":
            cifar10_transform = trn.Compose([trn.RandomHorizontalFlip(),
                                             trn.RandomCrop(32, padding=4),
                                             trn.ToTensor(),
                                             trn.Normalize(mean, std)])
        elif transform_mode == "test":
            cifar10_transform = trn.Compose([trn.ToTensor(),
                                             trn.Normalize(mean, std)])
        else:
            raise NotImplementedError

        if data_mode == "train":

            dataset = dset.CIFAR10(
                root=dataset_dir, train=True, download=True, transform=cifar10_transform)
        elif data_mode == "test":
            dataset = dset.CIFAR10(
                root=dataset_dir, train=False, download=True, transform=cifar10_transform)
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    return dataset


class ImageNetV2Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset_root = pathlib.Path(root)
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(
            self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label