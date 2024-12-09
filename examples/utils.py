import numpy as np
import os
import os
import pandas as pd
import pathlib
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv, SAGEConv
from tqdm import tqdm


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


def build_graph_dataset(dataset_name, device, ntrain_per_class=20, split_ratio=False):
    dataset_dir = get_dataset_dir()

    if dataset_name in ['cora_ml']:
        dataset = CitationFull(dataset_dir, dataset_name)
        graph_data = dataset[0].to(device)
        label_mask = F.one_hot(graph_data.y).bool()

        #######################################
        # training/validation/test data random split
        # 20 per class for training/validation, left for test
        #######################################

        if not split_ratio:
            classes_idx_set = [(graph_data.y == cls_val).nonzero(
                as_tuple=True)[0] for cls_val in graph_data.y.unique()]
            shuffled_classes = [
                s[torch.randperm(s.shape[0])] for s in classes_idx_set]

            train_idx = torch.concat([s[: ntrain_per_class]
                                      for s in shuffled_classes])
            val_idx = torch.concat(
                [s[ntrain_per_class: 2 * ntrain_per_class] for s in shuffled_classes])
            test_idx = torch.concat([s[2 * ntrain_per_class:]
                                     for s in shuffled_classes])
        else:
            num_nodes = graph_data.x.shape[0]
            rand_perm = torch.randperm(num_nodes)

            train_idx = rand_perm[:int(num_nodes * 0.2)].to(device)
            val_idx = rand_perm[int(num_nodes * 0.2):int(num_nodes * 0.3)].to(device)
            test_idx = rand_perm[int(num_nodes * 0.3):].to(device)

    else:
        raise NotImplementedError(
            f"The dataset {dataset_name} has not been implemented!")

    return graph_data, label_mask, train_idx, val_idx, test_idx


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, p_dropout):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
        self._p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self._p_dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, p_dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self._p_dropout = p_dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=self._p_dropout, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        device = x_all.device

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id]
                x = conv(x, batch.edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
            x_all = torch.cat(xs, dim=0).to(device)
        return x_all


def compute_adj_knn(features, k=20):
    features_normalized = features / features.norm(dim=1, keepdim=True)
    sims = torch.mm(features_normalized, features_normalized.t())
    sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0

    topk_values, topk_indices = torch.topk(sims, k, dim=1)

    adj_knn = torch.zeros_like(sims).to(features.device)
    rows = torch.arange(sims.shape[0]).unsqueeze(1).to(features.device)
    adj_knn[rows, topk_indices] = topk_values

    knn_edge = torch.nonzero(adj_knn).T
    knn_weights = adj_knn[knn_edge[0, :], knn_edge[1, :]]

    return knn_edge, knn_weights


from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import CitationFull, Amazon
import copy


def build_transductive_gnn_data(data_name, ntrain_per_class=20):
    data_dir = get_dataset_dir()

    if data_name in ['cora_ml']:
        graph_data = CitationFull(data_dir, data_name)[0]
        label_mask = F.one_hot(graph_data.y).bool()

        #######################################
        # training/validation/test data random split
        # ntrain_per_class per class for training/validation, left for test
        #######################################

        classes_idx_set = [(graph_data.y == cls_val).nonzero(
            as_tuple=True)[0] for cls_val in graph_data.y.unique()]
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

    return graph_data, label_mask, train_idx, val_idx, test_idx


def build_inductive_gnn_data(data_name, n_v=1000, n_t=10000, device='cuda:0'):
    data_dir = get_dataset_dir()

    if data_name in ['Computers']:
        graph_data = Amazon(data_dir, data_name,
                            pre_transform=RandomNodeSplit(split='train_rest', num_val=n_v, num_test=n_t))[0].to(device)
        kwargs = {'batch_size': 512, 'num_workers': 6,
                  'persistent_workers': True}
        train_loader = NeighborLoader(graph_data, input_nodes=graph_data.train_mask,
                                      num_neighbors=[25, 10], shuffle=True, **kwargs)
        subgraph_loader = NeighborLoader(copy.copy(graph_data), input_nodes=None,
                                         num_neighbors=[-1], shuffle=False, **kwargs)

        del subgraph_loader.data.x, subgraph_loader.data.y
        subgraph_loader.data.num_nodes = graph_data.num_nodes
        subgraph_loader.data.n_id = torch.arange(graph_data.num_nodes)
    else:
        raise NotImplementedError(
            f"The dataset {data_name} has not been implemented!")

    return graph_data, train_loader, subgraph_loader


def build_gnn_model(model_name):
    if model_name == "GCN":
        class GCN(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, p_dropout=0.8):
                super().__init__()
                self.conv1 = GCNConv(in_channels, hidden_channels, normalize=True)
                self.conv2 = GCNConv(hidden_channels, out_channels, normalize=True)
                self._p_dropout = p_dropout

            def forward(self, x, edge_index, edge_weight=None):
                x = self.conv1(x, edge_index, edge_weight).relu()
                x = F.dropout(x, p=self._p_dropout, training=self.training)
                x = self.conv2(x, edge_index, edge_weight)
                return x

        return GCN
    elif model_name == "SAGE":
        class SAGE(nn.Module):
            def __init__(self, in_channels, hidden_channels, out_channels, p_dropout=0.5):
                super().__init__()
                self.convs = torch.nn.ModuleList()
                self.convs.append(SAGEConv(in_channels, hidden_channels))
                self.convs.append(SAGEConv(hidden_channels, out_channels))

                self._p_dropout = p_dropout

            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                        x = F.dropout(x, p=self._p_dropout, training=self.training)
                return x

            @torch.no_grad()
            def inference(self, x_all, subgraph_loader):
                device = x_all.device

                # Compute representations of nodes layer by layer, using *all*
                # available edges. This leads to faster computation in contrast to
                # immediately computing the final representations of each batch:
                for i, conv in enumerate(self.convs):
                    xs = []
                    for batch in subgraph_loader:
                        x = x_all[batch.n_id]
                        x = conv(x, batch.edge_index)
                        if i < len(self.convs) - 1:
                            x = x.relu_()
                        xs.append(x[:batch.batch_size].cpu())
                    x_all = torch.cat(xs, dim=0).to(device)
                return x_all

        return SAGE
    else:
        raise NotImplementedError
