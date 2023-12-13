import os
import pathlib

from PIL import Image
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn
from torch.utils.data import Dataset, DataLoader


def build_dataset(dataset_name, transform = None, mode = "train"):

    
    #  path of usr
    usr_dir = os.path.expanduser('~')
    data_dir = os.path.join(usr_dir,"data")
    
    
    if dataset_name == 'imagenet': 
        if transform == None:
            transform = trn.Compose([
                            trn.Resize(256),
                            trn.CenterCrop(224),
                            trn.ToTensor(),
                            trn.Normalize(mean=[0.485, 0.456, 0.406],
                                        std =[0.229, 0.224, 0.225])
                            ])

        dataset = dset.ImageFolder(data_dir+"/imagenet/val", 
                                transform)
        num_classes = 1000  
    elif dataset_name == 'imagenetv2':
        if transform == None:
            transform = trn.Compose([
                            trn.Resize(256),
                            trn.CenterCrop(224),
                            trn.ToTensor(),
                            trn.Normalize(mean=[0.485, 0.456, 0.406],
                                        std =[0.229, 0.224, 0.225])
                            ])


        dataset = ImageNetV2Dataset(os.path.join(data_dir,"imagenetv2/imagenetv2-matched-frequency-format-val"),transform)

    elif dataset_name == 'mnist':
        if transform == None:
            transform = trn.Compose([
                        trn.ToTensor(),
                        trn.Normalize((0.1307,), (0.3081,))
            ])
        if mode ==  "train":
            dataset = dset.MNIST(data_dir, train=True, download=True, transform = transform)
        elif mode ==  "test":
            dataset = dset.MNIST(data_dir, train=False, download=True , transform = transform)

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
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label