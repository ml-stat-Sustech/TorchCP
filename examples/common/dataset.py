import numpy as np
import os
import pandas as pd
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.nn.functional as F
from torch_geometric.datasets import CitationFull


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
            dataset = dset.MNIST(data_dir, train=True,
                                 download=True, transform=transform)
        elif mode == "test":
            dataset = dset.MNIST(data_dir, train=False,
                                 download=True, transform=transform)
    else:
        raise NotImplementedError

    return dataset


base_path = "~/.cache/torchcp/datasets/"


def build_reg_data(data_name="community"):
    if data_name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(
            base_path + 'communities_attributes.csv', delim_whitespace=True)
        data = pd.read_csv(base_path + 'communities.data',
                           names=attrib['attributes'])
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
        y_wo_noise = 10 * np.sin(X[:, 0] * X[:, 1] * np.pi) + \
            20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
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


def build_gnn_data(data_name, ntrain_per_class=20):
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir, "data")

    if data_name in ['cora_ml']:
        dataset = CitationFull(data_dir, data_name)[0]
        label_mask = F.one_hot(dataset.y).bool()

        #######################################
        # training/validation/test data random split
        # ntrain_per_class per class for training/validation, left for test
        #######################################

        classes_idx_set = [(dataset.y == cls_val).nonzero(
            as_tuple=True)[0] for cls_val in dataset.y.unique()]
        shuffled_classes = [
            s[torch.randperm(s.shape[0])] for s in classes_idx_set]

        train_idx = torch.concat([s[: ntrain_per_class]
                                  for s in shuffled_classes])
        val_idx = torch.concat(
            [s[ntrain_per_class: 2 * ntrain_per_class] for s in shuffled_classes])
        test_idx = torch.concat([s[2 * ntrain_per_class:]
                                for s in shuffled_classes])
    else:
        raise NotImplementedError(
            f"The dataset {data_name} has not been implemented!")

    return dataset, label_mask, train_idx, val_idx, test_idx
