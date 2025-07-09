# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision
import torchvision.transforms as trn
from transformers import set_seed

from examples.utils import get_dataset_dir
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import LAC
from torchcp.classification.trainer import ConfTrTrainer

set_seed(seed=2025)

#######################################
# Preparing a calibration data and a test data
#######################################
transform = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean=[0.5071, 0.4867, 0.4408],
                  std=[0.2675, 0.2565, 0.2761])
])
dataset = torchvision.datasets.CIFAR100(
    root=get_dataset_dir(),
    train=False,
    download=True,
    transform=transform
)
train_dataset, cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [3000, 2000, 5000])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4,
                                              pin_memory=True)
test_instance, test_label = test_dataset[0]
test_instance = test_instance.unsqueeze(0)

#######################################
# Preparing a pytorch model
#######################################
init_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
init_model.to(device)
init_model.eval()

#######################################
# Conformal Training
#######################################

trainer = ConfTrTrainer(
    model=init_model,
    alpha=0.1,
    device=device
)

trained_model = trainer.train(train_dataloader, num_epochs=10)

########################################
# Conformal prediction
########################################

predictor = SplitPredictor(score_function=LAC(), model=trained_model, alpha=0.1)
predictor.calibrate(cal_dataloader)
predict_set = predictor.predict(test_instance)

