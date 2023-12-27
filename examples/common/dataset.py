import os
import pathlib

import numpy as np
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image
from torch.utils.data import Dataset


def build_dataset(dataset_name, transform=None, mode='train'):
    #  path of usr
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data")

    if dataset_name == 'imagenet':
        if transform is None:
            transform = trn.Compose([
                trn.Resize(256),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])

        dataset = dset.ImageFolder(data_dir + "/imagenet/val", transform)
    elif dataset_name == 'mnist':
        if transform is None:
            transform = trn.Compose([
                trn.ToTensor(),
                trn.Normalize((0.1307,), (0.3081,))
            ])
        if mode == "train":
            dataset = dset.MNIST(data_dir, train=True, download=True, transform=transform)
        elif mode == "test":
            dataset = dset.MNIST(data_dir, train=False, download=True, transform=transform)
    else:
        raise NotImplementedError

    return dataset


base_path = ".cache/data/"


def build_reg_data(data_name="community"):
    if data_name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', delim_whitespace=True)
        data = pd.read_csv(base_path + 'communities.data', names=attrib['attributes'])
        data = data.drop(columns=['state', 'county',
                                  'community', 'communityname',
                                  'fold'], axis=1)
        data = data.replace('?', np.nan)

        # Impute mean values for samples with missing values
        data['OtherPerCap'] = data['OtherPerCap'].astype("float")
        mean_value = data['OtherPerCap'].mean()
        data['OtherPerCap'].fillna(value=mean_value, inplace=True)
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values

    elif data_name == "synthetic":
        X = np.random.rand(500, 5)
        y_wo_noise = 10 * np.sin(X[:, 0] * X[:, 1] * np.pi) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
        eplison = np.zeros(500)
        phi = theta = 0.8
        delta_t_1 = np.random.randn()
        for i in range(1, 500):
            delta_t = np.random.randn()
            eplison[i] = phi * eplison[i - 1] + delta_t_1 + theta * delta_t
            delta_t_1 = delta_t

        y = y_wo_noise + eplison

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y



