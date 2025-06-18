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
from torchcp.classification.score import APS
from torchcp.classification.trainer import ConfTSTrainer

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
cal_dataset, conformal_cal_dataset, test_dataset = torch.utils.data.random_split(dataset, [3000, 2000, 5000])
cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
conformal_cal_dataloader = torch.utils.data.DataLoader(conformal_cal_dataset, batch_size=64, shuffle=False,
                                                       num_workers=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4,
                                              pin_memory=True)

#######################################
# Preparing a pytorch model
#######################################
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#######################################
# Temperature Scaling
#######################################
init_temperature = 1.0

trainer = ConfTSTrainer(
    model=model,
    init_temperature=init_temperature,
    alpha=0.1,
    device=device,
    verbose=True
)

calibrated_model = trainer.train(cal_dataloader, num_epochs=10)

########################################
# Conformal prediction
########################################

print("\nBefore Temperature Scaling:")
alpha = 0.1  # Significance level
predictor = SplitPredictor(score_function=APS(), model=model)
predictor.calibrate(conformal_cal_dataloader, alpha=0.1)
result_dict = predictor.evaluate(test_dataloader)
print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
print(f"Average Set Size: {result_dict['average_size']:.4f}")

print("\nAfter Temperature Scaling:")
result_dict_after = predictor.evaluate(test_dataloader)
predictor = SplitPredictor(score_function=APS(), model=calibrated_model)
predictor.calibrate(conformal_cal_dataloader, alpha=0.1)
result_dict = predictor.evaluate(test_dataloader)
print(f"Coverage Rate: {result_dict['coverage_rate']:.4f}")
print(f"Average Set Size: {result_dict['average_size']:.4f}")
