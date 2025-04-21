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
from torchcp.classification.score import THR
from torchcp.classification.trainer import SCPOTrainer

set_seed(seed=2025)

#######################################
# Preparing a calibration data and a test data
#######################################
transform = trn.Compose([
    trn.ToTensor(),
    trn.Normalize(mean=[0.4914, 0.4822, 0.4465],
                  std=[0.2470, 0.2435, 0.2616])
])
dataset = torchvision.datasets.CIFAR10(
    root=get_dataset_dir(),
    train=False,
    download=True,
    transform=transform
)

cal_dataset, conformal_cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [3000, 2000, 5000])
cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
conformal_cal_dataloader = torch.utils.data.DataLoader(conformal_cal_dataset, batch_size=256, shuffle=False,
                                                       num_workers=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4,
                                              pin_memory=True)

#######################################
# Preparing a pytorch model
#######################################
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#######################################
# Surrogate Conformal Predictor Optimization
#######################################

trainer = SCPOTrainer(
    alpha=0.01,
    model=model,
    device=device,
    verbose=True
)

calibrated_model = trainer.train(cal_dataloader, num_epochs=10)

########################################
# Conformal prediction
########################################

print("\nBefore Surrogate Conformal Predictor Optimization:")
predictor = SplitPredictor(score_function=THR(score_type="identity"), model=model)
predictor.calibrate(conformal_cal_dataloader, alpha=0.01)
result_dict = predictor.evaluate(test_dataloader)
print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
print(f"Average Set Size: {result_dict['average_size']:.4f}")

print("\nAfter Surrogate Conformal Predictor Optimization:")
predictor = SplitPredictor(score_function=THR(score_type="identity"), model=calibrated_model)
predictor.calibrate(conformal_cal_dataloader, alpha=0.01)
result_dict = predictor.evaluate(test_dataloader)
print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
print(f"Average Set Size: {result_dict['average_size']:.4f}")
